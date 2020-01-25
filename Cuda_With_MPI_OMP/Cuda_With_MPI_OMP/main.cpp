#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define KEEP_GOING 1
#define SHUTDOWN 0
#define MASTER_PROC 0
#define Q_FAILS 5
#define FAIL -1
#define OK_TAG 2
#define INPUT_FILE "C:\\data1.txt"
#define OUTPUT_FILE "C:\\output.txt"
#define MODE_LOCATION 0
#define MODE_VELOCITY 1

typedef struct
{
	int sign;
	double* location;
	double* velocity;
}Point;


// Function declerations
extern cudaError_t matchWithGPU(int* results, double* points, int* signs, double* w, int numOfPoints, int k);

extern cudaError_t updateLocationsWithGPU(double* locations, double* velocity, int numOfPoints, int k, double t);

void masterProc(double tMax, double dt, int numOfProcs, Point* arr, int numOfPoints, int k);

void slaveProcs(int numOfPoints, int k, int limit, double qc, double alpha, int* signs);

double checkAllPoints(double* points, double * w, int* signs, int size, int limit, double qc, double alpha, int k);

void shutdownLoop(int startAfter, int numOfProcs);

double* updateLocations(Point* arr, int numOfPoints, double t, int k);

void updateW(double* w, int value, double* p, int offset, double alpha, int k);

double f(double* p, int offset, double* w, int k);

Point* readFromFile(int* numOfPoints, int* k, double* dt, double* tMax, double* alpha, int* limit, double* qc);

void checkAllocation(const void* p);

double* flattenPoints(Point* arr, int numOfPoints, int k, int mode);

int* getSigns(Point* arr, int numOfPoints);

void freePoints(Point* points, int numOfPoints);

void freeWeights(double** weights, int size);

void saveToFile(double bestTime, double bestQ, double* w, int k, int success);

void setArrayTo(double* arr, int size, int info);

double** initWeightsResults(int slaves, int k);

int getBestResult(double* tResult, double* qResult, int numOfSlaves);

int sign(double f);

double matchPoints(double* points, int* signs, double* w, int numOfPoints, int k);

int sumResults(int* results, int numOfPoints);

// implementations

/*
	Complexity assumption:
	In all methods there is an unlimited amount of resources(Processes, threads)
*/


//Main function. 
//COMPLEXITY : O( O(MasterProc()) + O(SlaveProc()) + O(readFromFile()) ~~~O((k*N*limit)). Without parallelizing -> O(k*N*limit*t)
int main(int argc, char* argv[])
{
	int myId, numOfProcs;
	int numOfPoints;
	Point* points;
	double tMax, dt, qc, alpha;
	int k, limit;
	int* signsArr;

	//initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcs);

	//In master-slave, there needs to be more than two processes
	if (numOfProcs < 2)
	{
		printf("Not enough processes. There should be at least two.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (myId == MASTER_PROC) //Master reads from file, and derives signsArr from points array
	{
		points = readFromFile(&numOfPoints, &k, &dt, &tMax, &alpha, &limit, &qc);
		signsArr = getSigns(points, numOfPoints);
	}

	//broadcast elements relevant to all slaves
	MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
	if (myId != MASTER_PROC) //Slaves need to create space to recieve signsArr
	{
		signsArr = (int*)calloc(numOfPoints, sizeof(int));
		checkAllocation(signsArr);
	}
	MPI_Bcast(&k, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
	MPI_Bcast(&limit, 1, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);
	MPI_Bcast(&qc, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);
	MPI_Bcast(&alpha, 1, MPI_DOUBLE, MASTER_PROC, MPI_COMM_WORLD);
	MPI_Bcast(signsArr, numOfPoints, MPI_INT, MASTER_PROC, MPI_COMM_WORLD);

	if (myId == MASTER_PROC)
		masterProc(tMax, dt, numOfProcs, points, numOfPoints, k);
	else
		slaveProcs(numOfPoints, k, limit, qc, alpha, signsArr);

	if (myId == MASTER_PROC)
		freePoints(points, numOfPoints);
	free(signsArr);

	MPI_Finalize();
	return 0;
}

//master proc comes here
//COMPLEXITY: MPI function. O(t*O(updateLocations())) ~~~ O(t). Without parallelizing -> O(t*N*k)
void masterProc(double tMax, double dt, int numOfProcs, Point* arr, int numOfPoints, int k)
{
	MPI_Status status;
	double t = 0;
	int i;
	int loop = KEEP_GOING;
	double* tResult;
	double* qResult;
	int activeSlaves = numOfProcs - 1;
	double* pointsLocationsArr = 0;
	double** weightsResults;
	int bestResult = FAIL;

	//Init weights matrix for results from slaves
	weightsResults = initWeightsResults(activeSlaves, k);

	//create arrays for times and q results from slaves
	tResult = (double*)calloc(activeSlaves, sizeof(double));
	checkAllocation(tResult);
	qResult = (double*)calloc(activeSlaves, sizeof(double));
	checkAllocation(qResult);

	setArrayTo(qResult, activeSlaves, FAIL); //set the array of Qs to FAIL

	while (t < tMax)
	{
		//Send loop. Sends to each slave the current time and current points location, then updates them.
		for (i = 1; i < numOfProcs; i++)
		{
			//When reaching this code line, either in the first iteration or any other, all the slaves are waiting to recieve.
			//hence sending to slave i ensures that slave i is waiting
			MPI_Send(&t, 1, MPI_DOUBLE, i, KEEP_GOING, MPI_COMM_WORLD);
			t = t + dt;

			if (!pointsLocationsArr) //On first iteration, the array isn't allocated
				pointsLocationsArr = flattenPoints(arr, numOfPoints, k, MODE_LOCATION); //Flattens the points array to one dimension.
			MPI_Send(pointsLocationsArr, numOfPoints * k, MPI_DOUBLE, i, KEEP_GOING, MPI_COMM_WORLD);
			free(pointsLocationsArr); //After sending the array to the slave, the current locations array is no longer relevant.

			pointsLocationsArr = updateLocations(arr, numOfPoints, t, k);  //updates locations, as a byproduct, also returns the new locations array

			if (t >= tMax) //shutting down slaves that may wait for their turn after tMax was achieved.
			{
				printf("Shutting down %d unneccessary slaves\n", activeSlaves - i);
				shutdownLoop(i - 1, numOfProcs);
				activeSlaves = i;
				loop = SHUTDOWN; //since tMax was achieved, no need to send anymore data.
				break;
			}
		}
		for (i = 0; i < activeSlaves; i++) //Recieve section
		{
			MPI_Recv(&qResult[i], 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == OK_TAG) //q results are always recieved. If q is valid, expect to recieve time and weights
			{
				MPI_Recv(&tResult[i], 1, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(weightsResults[i], k, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				loop = SHUTDOWN; //If even one q result was valid, there's no need to continue checking times.
								// but still want to recieve results, as the next slaves may have earlier time frames.
			}
		}

		if (loop == SHUTDOWN) //Enter here only after a result was found or all timeframes have passed.
		{
			bestResult = getBestResult(qResult, tResult, numOfProcs - 1); //finds the time iteration with the earliest timeframe that has a valid q result.
			if (bestResult == FAIL) //If no q was valid, notify the user.
			{
				printf("Time was not found.\n");
				break;
			}
			printf("The first timeframe to succeed is %lf with Q value of %lf\n", tResult[bestResult], qResult[bestResult]);
			printf("With weights:\n");
			for (i = 0; i < k; i++)
			{
				printf("w%d = %lf\n", i, weightsResults[bestResult][i]);
			}
			break;
		}
	}

	/*
		Send shutdown signal to all slaves.
		- In case a result was found all slaves are still active and needs to be shut down.
		- In case tmax was reached, all slaves before that time are still active.
	*/
	shutdownLoop(0, numOfProcs);

	//saves to file the results OR, if bestResult is a fail, save failure.
	printf("Note: Results are also written to file %s\n", OUTPUT_FILE);
	saveToFile(tResult[bestResult], qResult[bestResult], weightsResults[bestResult], k, bestResult);

	freeWeights(weightsResults, numOfProcs - 1);
	free(tResult);
	free(qResult);

}

//slaves come here
//COMPLEXITY: MPI function. O(O(checkAllPoints()) ~~~ O(k*N*limit). Without parallelizing -> O(k*N*limit*t)
void slaveProcs(int numOfPoints, int k, int limit, double qc, double alpha, int* signs)
{
	MPI_Status status;
	double myTime;
	double q;
	int tag = Q_FAILS;
	double* points;
	int flag = KEEP_GOING;
	double* w;

	//initalize weights array and allocate space for points
	w = (double*)calloc(k, sizeof(double));
	checkAllocation(w);
	points = (double*)calloc(numOfPoints * k, sizeof(double));
	checkAllocation(points);

	while (status.MPI_TAG != SHUTDOWN) //Keep looping until a shutdown signal is recieved.
	{
		/*
		Recieve the time the slave is checking
		OR
		Recieve a shutdown signal with junk in myTime
		*/
		MPI_Recv(&myTime, 1, MPI_DOUBLE, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == SHUTDOWN)
			break;

		MPI_Recv(points, (numOfPoints * k), MPI_DOUBLE, MASTER_PROC, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		setArrayTo(w, k, 0); // reset W vector to 0

		//checks all the points according to the algorithm, returns error margin 
		q = checkAllPoints(points, w, signs, numOfPoints, limit, qc, alpha, k);

		//Every slave sends q regardless. If q is valid, proceed to also send the time and weights
		if (q != FAIL)
			tag = OK_TAG;
		MPI_Send(&q, 1, MPI_DOUBLE, MASTER_PROC, tag, MPI_COMM_WORLD);
		if (q != FAIL)
		{
			MPI_Send(&myTime, 1, MPI_DOUBLE, MASTER_PROC, tag, MPI_COMM_WORLD);
			MPI_Send(w, k, MPI_DOUBLE, MASTER_PROC, tag, MPI_COMM_WORLD);
		}
	} // end of while

	free(w);
	free(points);
}

//checks all points and update weights vector according to the algorithm.
//COMPLEXITY: Iterative function. O(k*N*limit + O(matchAllPoints)) ~~~ o(k*N*limit)
//Note: This function cannot be parallelized as a problematic point would require it to stop, fix itself and continue from that place.
double checkAllPoints(double* points, double * w, int* signs, int numOfPoints, int limit, double qc, double alpha, int k)
{
	int i;
	int value = 0;
	int iterations = 0;
	double q = FAIL;

	for (iterations = 0; iterations < limit; iterations++) //The algorithm only have a limited amount of iterations to improve itself.
	{
		{
			for (i = 0; i < numOfPoints; i++)
			{
				value = sign(f(points, i*k, w, k));
				if (value != signs[i]) //update W on every point that fails the test, checking them iteratively
					//if the test failed, update weights vector.
					updateW(w, -value, points, i*k, alpha, k);
			}
			//After going over all points, check all points with the current weights vector to find q = error margin
			q = matchPoints(points, signs, w, numOfPoints, k);
			if (q == 0) // if there's a perfect match, stop
				break;
		}

		if (q < qc && q >= 0)
			return q;
		else
			return FAIL;

	}
}
//checks all points with a final weights vector, calculate error margin
//COMPLEXITY: N CUDA threads do O(k) each, without parallelizing -> O(k*N)
double matchPoints(double* points, int* signs, double* w, int numOfPoints, int k)
{
	int pMiss = 0;
	int* results;

	results = (int*)calloc(numOfPoints, sizeof(int));
	checkAllocation(results);

	//using CUDA, raise flags in results vector if a point fails
	cudaError_t cudaStatus = matchWithGPU(results, points, signs, w, numOfPoints, k);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "mathWithGPU failed\n");
		exit(1);
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		exit(1);
	}

	//using OMP, reduce the results array to pMiss
	pMiss = sumResults(results, numOfPoints);

	free(results);

	//q = pMiss/numOfPoints
	return (double)pMiss / (double)numOfPoints;
}

//Using OMP, reduce the results vector into one variable.
//COMPLEXITY: N threads do o(1) each, without parallelizing -> O(N)
int sumResults(int* results, int numOfPoints)
{
	int pMiss = 0;
	int i;

#pragma omp parallel for reduction(+ : pMiss)
	for (i = 0; i < numOfPoints; i++)
		pMiss += results[i];

	return pMiss;
}

//sends shutdown signal to all slaves after the given number
//COMPLEXITY: Iterative function. O(nProcs)
void shutdownLoop(int startAfter, int numOfProcs)
{
	int i;
	double garbageData = 1;
	for (i = startAfter + 1; i < numOfProcs; i++)
	{
		MPI_Send(&garbageData, 1, MPI_DOUBLE, i, SHUTDOWN, MPI_COMM_WORLD);
	}
}

//updates locations of points based on velocity and time
//COMPLEXITY: N*k CUDA threads do O(1) each, without parallelizing -> O(N*k)
double* updateLocations(Point* arr, int numOfPoints, double t, int k)
{
	double* locations;
	double* velocity;
	cudaError_t cudaStatus;

	//flattens the location and velocity vectors over k dimensions to a 1d array
	locations = flattenPoints(arr, numOfPoints, k, MODE_LOCATION);
	velocity = flattenPoints(arr, numOfPoints, k, MODE_VELOCITY);

	//Using CUDA, calculate P = P0 + V*t, whereas P is the location
	cudaStatus = updateLocationsWithGPU(locations, velocity, numOfPoints, k, t);

	//the flattened velocity vector isn't needed anymore.
	free(velocity);

	//Locations vector can be used outside the function
	return locations;
}

//Using OMP, calculate weights based on the algorithm
//COMPLEXITY: k threads do o(1) each, without parallelizing -> O(k)
void updateW(double* w, int value, double* p, int offset, double alpha, int k)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < k; i++)
		w[i] = w[i] + p[offset + i] * (alpha * value);
}

//Using omp reduction, calculate function f based on the algorithm
//COMPLEXITY: k threads do o(1) each,  without parallelizing -> O(k)
double f(double* p, int offset, double* w, int k)
{
	int i;
	double result = 0;

#pragma omp parallel for reduction(+ : result)
	for (i = 0; i < k; i++)
		result += w[i] * p[offset + i];

	return result;
}

//reads everything from the input file
//COMPLEXITY: Iterative function. O(N)
Point* readFromFile(int* numOfPoints, int* k, double* dt, double* tMax, double* alpha, int* limit, double* qc)
{
	FILE* fp;
	Point* arr;
	int i, j;

	fopen_s(&fp, INPUT_FILE, "r");
	if (!fp)
		fclose(fp);
	checkAllocation(fp);

	fscanf_s(fp, "%d", numOfPoints);
	fscanf_s(fp, "%d", k);
	*k = *k + 1;  // K + 1 to match W0
	fscanf_s(fp, "%lf", dt);
	fscanf_s(fp, "%lf", tMax);
	fscanf_s(fp, "%lf", alpha);
	fscanf_s(fp, "%d", limit);
	fscanf_s(fp, "%lf", qc);

	arr = (Point*)calloc(*numOfPoints, sizeof(Point));
	if (!arr)
		fclose(fp);
	checkAllocation(arr);

	for (i = 0; i < *numOfPoints; i++)
	{
		arr[i].location = (double*)calloc(*k, sizeof(double));
		checkAllocation(arr[i].location);
		arr[i].location[*k - 1] = 1; //last element is 1 on every point
		for (j = 0; j < (*k - 1); j++)
		{
			fscanf_s(fp, "%lf", &arr[i].location[j]);
		}

		arr[i].velocity = (double*)calloc(*k, sizeof(double));
		checkAllocation(arr[i].velocity);
		arr[i].velocity[*k - 1] = 0; //last element never moves.
		for (j = 0; j < (*k - 1); j++)
		{
			fscanf_s(fp, "%lf", &arr[i].velocity[j]);
		}
		fscanf_s(fp, "%d", &arr[i].sign);
	}
	fclose(fp);
	return arr;
}

//checks if a dynamic allocation succeeded
//COMPLEXITY: Iterative function. O(1)
void checkAllocation(const void* p)
{
	if (!p)
	{
		printf("Allocation failed.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}

//Flattens the locations/velocity of all points in k dimensions to a 1d array using OMP
//COMPLEXITY: N threads do o(k) each, without parallelizing -> O(k*N)
double* flattenPoints(Point* arr, int numOfPoints, int k, int mode)
{
	int i, j = 0;
	double* flatArr;

	flatArr = (double*)calloc(numOfPoints * k, sizeof(double));
	checkAllocation(flatArr);

#pragma omp parallel for private(j)
	for (i = 0; i < numOfPoints; i++)
	{
		for (j = 0; j < k; j++)
		{
			if (mode == MODE_LOCATION)
				flatArr[i*k + j] = arr[i].location[j];
			else
				flatArr[i*k + j] = arr[i].velocity[j];
		}
	}

	return flatArr;
}

//extracts signs of all points to one array using OMP
//COMPLEXITY: N threads do o(1) each,  without parallelizing -> O(N)
int* getSigns(Point* arr, int numOfPoints)
{
	int i;
	int* signs = (int*)calloc(numOfPoints, sizeof(int));
	checkAllocation(signs);

#pragma omp parallel for
	for (i = 0; i < numOfPoints; i++)
	{
		signs[i] = arr[i].sign;
	}

	return signs;
}

//saves results to output file
//COMPLEXITY: Iterative function. O(k)
void saveToFile(double bestTime, double bestQ, double* w, int k, int success)
{
	FILE* fp;
	int i;
	fopen_s(&fp, OUTPUT_FILE, "w");
	if (!fp)
		fclose(fp);
	checkAllocation(fp);

	if (success == FAIL)
	{
		fprintf_s(fp, "Time was not found\n");
		fclose(fp);
		return;
	}

	fprintf_s(fp, "Minimum t = %lf q = %lf\n", bestTime, bestQ);
	for (i = 0; i < k; i++)
	{
		fprintf_s(fp, "%lf\n", w[i]);
	}

	fclose(fp);
}

//Using OMP, set all elements in a given array to the given value
//COMPLEXITY: size threads do o(1), without parallelizing -> O(size). 
void setArrayTo(double* arr, int size, int value)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < size; i++)
		arr[i] = value;
}

//frees the points array using OMP
//COMPLEXITY: N threads do O(1), without parallelizing -> O(N)
void freePoints(Point* points, int numOfPoints)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < numOfPoints; i++)
	{
		free(points[i].location);
		free(points[i].velocity);
	}
	free(points);
}

//frees the weights 2d array
//COMPLEXITY: k threads do o(1), without parallelizing -> O(k)
void freeWeights(double** weights, int k)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < k; i++)
		free(weights[i]);
	free(weights);
}

//Initialize a 2d weights array
//COMPLEXITY: Iterative function. O(nProcs)
double** initWeightsResults(int slaves, int k)
{
	int i;
	double** weightsResult = (double**)calloc(slaves, sizeof(double*));
	checkAllocation(weightsResult);
	for (i = 0; i < slaves; i++)
	{
		weightsResult[i] = (double*)calloc(k, sizeof(double));
		checkAllocation(weightsResult[i]);
	}

	return weightsResult;
}

//Calculates the best result
//A result is considered optimal if it has the minimal time of all valid q's
//COMPLEXITY: Iterative function. O(nProcs)
int getBestResult(double* qResult, double* tResult, int numOfSlaves)
{
	int i;
	int bestResult = -1;
	double tMin = -1;

	for (i = 0; i < numOfSlaves; i++) //get the first position where q isn't a failure
	{
		if (qResult[i] != FAIL)
		{
			bestResult = i;
			tMin = tResult[i];
			break;
		}
	}

	for (; i < numOfSlaves; i++) //keep looping, find a q that isn't a failure AND has an earlier time than previous best result.
		if (tResult[i] < tMin && qResult[i] != FAIL)
		{
			tMin = tResult[i];
			bestResult = i;
		}

	return bestResult;
}

//returns the sign(+-) of a given double
//COMPLEXITY: Iterative function. O(1)
int sign(double f)
{
	if (f < 0)
		return -1;
	return 1;
}
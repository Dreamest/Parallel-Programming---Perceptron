
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define MAX_THREADS 1024

//declerations
cudaError_t matchWithGPU(int* results, double* points, int* signs, double* w, int numOfPoints, int k);

cudaError_t updateLocationsWithGPU(double* locations, double* velocity, int numOfPoints, int k, double t);

void checkError_locations(cudaError_t cudaStatus, double* dev_locations, double* dev_velocity, const char* errorMessage);

void freeCudaMemory_locations(double* dev_locations, double* dev_velocity);

void freeCudaMemory_match(int* dev_results, double* dev_points, int* dev_signs, double* dev_w);

void checkError_match(cudaError_t cudaStatus, int* dev_results, double* dev_points, int* dev_signs, double* dev_w, const char* errorMessage);

__device__ int signGPU(double f);

__device__ double fGPU(const double* p, int offset, const double* w, int k);


// implmentations

//changes results at a given index to 1 if the test fails
__global__ void matchKernel(int* results, const double* points, const int* signs, const double* w, int numOfPoints, int k)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x; //calculate index of each element in array
	int value = signGPU(fGPU(points, index*k, w, k));

	if (value != signs[index])
		results[index] = 1;
}

//updates the location at index based on location=location0 +velocity * t
__global__ void updateLocationsKernel(double* location, const double* velocity, double t)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x; //calculate index of each element in array

	location[index] = location[index] + velocity[index] * t;
}

//function to free all arrays related to the match part of CUDA
void freeCudaMemory_match(int* dev_results, double* dev_points, int* dev_signs, double* dev_w)
{
	cudaFree(dev_results);
	cudaFree(dev_points);
	cudaFree(dev_signs);
	cudaFree(dev_w);
}

//function responsible to handle errors in the match part of CUDA
void checkError_match(cudaError_t cudaStatus, int* dev_results, double* dev_points, int* dev_signs, double* dev_w, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		freeCudaMemory_match(dev_results, dev_points, dev_signs, dev_w);
	}
}

//tests all points to see if they're in the place they should be
cudaError_t matchWithGPU(int* results, double* points, int* signs, double* w, int numOfPoints, int k)
{
	char errorBuffer[100];
	int* dev_results = 0;
	double* dev_points = 0;
	int* dev_signs = 0;
	double* dev_w = 0;
	int extra = 0;
	int numOfBlocks, numOfThreads;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Allocate GPU buffers for every array
	cudaStatus = cudaMalloc((void**)&dev_results, numOfPoints * sizeof(int));
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_points, numOfPoints * k * sizeof(double));
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_signs, numOfPoints * sizeof(int));
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_w, k * sizeof(double));
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMalloc failed!");

	// Copy input arrays from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, numOfPoints * k * sizeof(double), cudaMemcpyHostToDevice);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_signs, signs, numOfPoints * sizeof(int), cudaMemcpyHostToDevice);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_w, w, k * sizeof(double), cudaMemcpyHostToDevice);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMemcpy failed!");

	// Calculate the number of blocks and threads needed.
	extra = numOfPoints % MAX_THREADS != 0 ? 1 : 0;
	numOfBlocks = (numOfPoints / MAX_THREADS + extra);
	numOfThreads = MAX_THREADS>numOfPoints ? numOfPoints : MAX_THREADS;

	// Launch a kernel on the GPU with one thread for each element.
	matchKernel<<<numOfBlocks, numOfThreads >>>(dev_results, dev_points, dev_signs, dev_w, numOfPoints, k);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "matchKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, errorBuffer);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, errorBuffer);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, dev_results, numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	checkError_match(cudaStatus, dev_results, dev_points, dev_signs, dev_w, "cudaMemcpy failed!");

	freeCudaMemory_match(dev_results, dev_points, dev_signs, dev_w);

	return cudaStatus;
}

//checks the sign of a given double
__device__ int signGPU(double f)
{
	if (f < 0)
		return -1;
	return 1;
}

//calculates f based on the algorithm
__device__ double fGPU(const double* p, int offset, const double* w, int k)
{
	int i;
	double result = 0;

	for (i = 0; i < k; i++)
	{
		result += w[i] * p[offset + i];
	}

	return result;
}

//updates the locations vector to its new position based on time and velocity
cudaError_t updateLocationsWithGPU(double* locations, double* velocity, int numOfPoints, int k, double t)
{
	char errorBuffer[100];
	double* dev_locations = 0;
	double* dev_velocity = 0;
	int extra;
	int numOfBlocks, numOfThreads;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	// Allocate GPU buffers for the arrays
	cudaStatus = cudaMalloc((void**)&dev_locations, numOfPoints * k * sizeof(double));
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&dev_velocity, numOfPoints * k * sizeof(double));
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMalloc failed!");

	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_locations, locations, numOfPoints * k * sizeof(double), cudaMemcpyHostToDevice);
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(dev_velocity, velocity, numOfPoints * k * sizeof(double), cudaMemcpyHostToDevice);
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMemcpy failed!");

	// Calculate the number of blocks and threads needed.
	extra = (numOfPoints*k) % MAX_THREADS != 0 ? 1 : 0;
	numOfBlocks = ((numOfPoints*k) / MAX_THREADS + extra);
	numOfThreads = MAX_THREADS>(numOfPoints*k) ? (numOfPoints*k) : MAX_THREADS;

	// Launch a kernel on the GPU with one thread for each element. Each calculating a dimension in a point, going over all points
	updateLocationsKernel<<<numOfBlocks, numOfThreads>>>(dev_locations, dev_velocity, t);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMemcpy failed!");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMemcpy failed!");

	// Copy output vector from GPU buffer to host memory.
	// In this case, overwrite the locations array on host memory with the updated one
	cudaStatus = cudaMemcpy(locations, dev_locations, numOfPoints * k * sizeof(double), cudaMemcpyDeviceToHost);
	checkError_locations(cudaStatus, dev_locations, dev_velocity, "cudaMemcpy failed!");

	freeCudaMemory_locations(dev_locations, dev_velocity);

	return cudaStatus;
}

//function responsible to handle errors in the update locations part of CUDA
void checkError_locations(cudaError_t cudaStatus, double* dev_locations, double* dev_velocity, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		freeCudaMemory_locations(dev_locations, dev_velocity);
	}
}

//function responsible to handle errors in the update locations part of CUDA
void freeCudaMemory_locations(double* dev_locations, double* dev_velocity)
{
	cudaFree(dev_locations);
	cudaFree(dev_velocity);
}
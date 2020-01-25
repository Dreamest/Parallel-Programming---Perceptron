# Parallel-Programming---Perceptron

Problem Definition
  Given a set of N points in K-dimensional space. Each point X is marked as belonging to set A or B. Implement a Simplified Binary Classification algorithm to find a Linear Classifier. The result depends on the maximum iteration allowed, value of the chosen parameter alpha and the time value t. The purpose of the project is to define a minimal value of t that leads to the Classifier with acceptable value of Quality of Classifier.

The Perceptron Classifier
  Given linearly seperable data Xi labelled into two categories Yi = {-1, 1}, need to find a weight vector W such that the discriminant  function f(Xi) = WXi + W0  separates the categories.

  Cycle through the data points {Xi, Yi} 
    If Xi is misclassified then Wi = Wi + alpha*sign(f(Xi))*Xi
  Repeat until all the data is correctly classified or maximum iterations achieved. 

Input Data:

  N - Number of points
  
  K - Number of coordinates of a point
  
  dT - Increment value of t
  
  TMax - maximum value of t
  
  alpha - conversion ratio
  
  LIMIT - the maximum number of iterations
  
  QC - Acceptable level of quality from the classifier
  
  The points 
  
    - location in k dimensions
    
    - speed in k dimensions
    
  - point label Y = {-1, 1}
  
Output Data:
  Minimum t - The earliest timeframe in which a valid result was achieved
  q - the quality of the classifier at that time
  W0
  .
  .
  .
  Wk

Solution: 
  See flowchart PDF for simplified explanation of the methodology.

  The solution was compiled in Visual Studio 2015.
  Bias weight is at Wk
  
  Using 3 Parallelization tools: MPI, OMP and CUDA
    - MPI was used to parallel the timeframes, as MPI uses processes, it should be given the largest portion of the computing power.
    - OMP was used for small tasks, like summing arrays
    - CUDA was used for manipulations that required going through the entire dataset, was used to update the points' locations and to estimate the perceptron quality when needed.
  
  

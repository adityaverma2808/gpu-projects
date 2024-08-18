#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

using namespace std;

//*******************************************

// functors and device functions ===============================================

struct ThresholdToBinary {
  __device__
  int operator()(const int& value) const 
  {
    if(value > 0)   return 1;
    return 0;
  }
};


__device__ int distancecalc(int x1 ,int y1 ,int x2 ,int y2)
{
    int a1 = x1;
    int b1=y1;
    int a2=x2;
    int b2=y2;
    int dist1 = (a1-a2);
    int dist2 = (b1-b2);
    if(dist1 < 0) dist1 = dist1 * -1;
    if(dist2 < 0) dist2 = dist2 * -1;
    return dist1+dist2;
}
__device__ bool check_collinear(int x1 ,int y1 ,int x2 ,int y2 ,int x3 ,int y3)
{
  long long int a1 = x1;
  long long int b1=y1;
  long long int a2=x2;
  long long int b2=y2;
  long long int a3=x3;
  long long int b3=y3;
  long long int first = a1 * (b2 - b3);
  long long int second = a2 * (b3 - b1);
  long long int third = a3 * (b1 - b2);
  long long int az = first + second + third;
  if(az == 0) return 1;
  return 0;


}


__device__ unsigned int closest_power_of_2(unsigned int val) 
{
    
    val = val-1;
    val = val | (val >> 1);
    val = val | (val >> 2);
    val = val | (val >> 4);
    val = val | (val >> 8);
    val = val | (val >> 16);
    val = val+1;
    return val;
}

__device__ int direction(int x1 ,int y1 ,int x2 ,int y2)
{
    int quadrant;
    if (x1 >= x2)
    {
        if (y1 > y2)    return 1;
        return 2;
    }
    if (y1 > y2)    return 3;
    return 4;
}

// kernels ===================================================


__global__
void target_setting(int* tankdir ,int *xcoord,int *ycoord,int* HP , int T , int i)
{
    int minimumDistences=0;
    __shared__ int distance_matrix[1024];
    minimumDistences=1;
    int tankid = blockIdx.x;
    int target = (tankid+i)%T;
    int inlinetank = threadIdx.x;
    distance_matrix[inlinetank] = INT_MAX;
    int x1 = xcoord[tankid] ;
    int y1 = ycoord[tankid] ;
    int x2 = xcoord[target] ;
    int y2 = ycoord[target] ;
    int x3 = xcoord[inlinetank] ;
    int y3 = ycoord[inlinetank];
    int distid = INT_MAX;


    // filtering the valid tanks and calculating their distance

    if(tankid != inlinetank && HP[inlinetank]>0 )
    {
        if(check_collinear(x1,y1,x2,y2,x3,y3) && (direction(x1,y1,x2,y2) == direction(x1,y1,x3,y3)))
        {

            distance_matrix[inlinetank] = distancecalc(x1,y1,x3,y3);
            target = inlinetank;
            distid = distancecalc(x1,y1,x3,y3);
            minimumDistences = distid;
        }

          
    }
    
    __syncthreads();


  int T_pad = closest_power_of_2(T);
   int off = T_pad/2;
  for( ; off ; off = off/ 2)
  {
      
        int currentTank = inlinetank + off;
        if(inlinetank < off && currentTank >= T)
        {
            atomicMin(&distance_matrix[inlinetank],INT_MAX);
        }
        else if(inlinetank < off)
        {
            atomicMin(&distance_matrix[inlinetank],distance_matrix[inlinetank+off]);

        }
        
      __syncthreads();

  }

    int distanceValue = distance_matrix[0];
    if(distanceValue == INT_MAX) tankdir[tankid] = -1;
    else if(distanceValue  ==  distid) tankdir[tankid] = inlinetank;


}

__global__
void updatescore(int *score,int* HP ,int* tankdir)
{

  if(HP[threadIdx.x] <= 0) tankdir[threadIdx.x] = -1;
    int tankconf = tankdir[threadIdx.x];
  if(tankconf != -1 )
  {
    atomicAdd(&score[threadIdx.x],1);
    atomicAdd(&HP[tankdir[threadIdx.x]],-1);
  }

}


// Write down the kernels here


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;


    FILE *inputfilepointer;

    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }


    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

// initial setup ===============================================================
  int i=0;
  thrust::device_vector<int> HP(T,H);
  thrust::device_vector<int> tankdir(T);
   int* tankdir_ptr = thrust::raw_pointer_cast(tankdir.data());
  int* HP_ptr = thrust::raw_pointer_cast(HP.data());

  int *xcoord_d ;
  cudaMalloc(&xcoord_d , sizeof(int)*T);
  cudaMemcpy(xcoord_d, xcoord, sizeof(int)*T, cudaMemcpyHostToDevice);

  int *ycoord_d;
  cudaMalloc(&ycoord_d , sizeof(int)*T);
  cudaMemcpy(ycoord_d, ycoord, sizeof(int)*T, cudaMemcpyHostToDevice);

  int  *score_d;


  // allocating cuda memory and copying it

  
  cudaMalloc(&score_d , sizeof(int)*T);

  
  int sum = thrust::transform_reduce(HP.begin(), HP.end(), ThresholdToBinary(), 0, thrust::plus<int>());
  
  int mini = T;
 

  while(sum >1 && (i = i+1))
  {
      if(i % T ){

        target_setting<<<T,T>>>(tankdir_ptr,xcoord_d,ycoord_d,HP_ptr,T,i);
        mini = i;
        updatescore<<<1,T>>>(score_d, HP_ptr , tankdir_ptr );

        // Reduce to get the sum of transformed elements
        auto start_itr =  HP.begin();
        sum = thrust::transform_reduce(start_itr, HP.end(), ThresholdToBinary(), 0, thrust::plus<int>());
        cudaDeviceSynchronize();

      }

  }
    i=0;
  cudaMemcpy(score, score_d, sizeof(int)*T, cudaMemcpyDeviceToHost);
    cudaFree(xcoord_d);
    cudaFree(ycoord_d);



    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
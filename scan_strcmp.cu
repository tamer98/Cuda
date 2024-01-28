#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024 // number of threads in a block

/* Here we do an inclusive scan of 'array' in place.
   'size' is the number of elements in 'array'.
   it should be a power of 2.
 
   We assume that 'array' is in shared memory so that there is no need to 
   copy it to shared memory here.
    */
__device__ void scan_plus(int *array, int size)
{
   for (unsigned int stride=1; stride <= size/2; stride *= 2) {
        int v;
        if (threadIdx.x >= stride) {
            v = array[threadIdx.x - stride];
        }
        __syncthreads(); /* wait until all threads finish reading 
		                    an element */

        if (threadIdx.x >= stride)
            array[threadIdx.x] += v;

        __syncthreads(); /* wait until all threads finish updating an
		                    element */
     }
     
} // scan_plus

/*
   This kernel compares the two strings s1 and s2. Both strings are
   terminated with a null byte.
   The result is an integer:  0, if s1 and s2 are equal;
                              a negative value if s1 is less than s2;
                              a positive value if s1 is greater than s2
   The argument 'result' is used to "return" the result.
   The arguments n1, n2 indicate the number of characters in s1 and s2, respectively
    (including the null byte at the end).
             
   We assume that the number of threads in a block is >= max(n1,n2).  
*/
__global__ void my_strcmp(const char  *s1, int n1, const char *s2, int n2,  int *result)
{


 __shared__ int differentValues[BLOCK_DIM];
    int threadID = threadIdx.x;
    int stepSize = blockDim.x;
    int localDifference = 0;


	for (int index = threadID; index < min(n1, n2); index += stepSize) 
	{
   		 if (s1[index] != s2[index]) 
   		 {
   		 
       		   localDifference = s1[index] - s2[index];
       		   break;
       		   
    	}
    		
    }


  differentValues[threadID] = localDifference;
  __syncthreads();



  scan_plus(differentValues, BLOCK_DIM);
  __syncthreads();

  if (threadID == 0)
   {
   
     *result = differentValues[BLOCK_DIM - 1];
     
   }


}


int main(int argc, char **argv) 
{

	char *dev_s1, *dev_s2;
    int *dev_result;
#if 0
    char s1[] = "supercalifragilisticexpialidocious";
    char s2[] = "supercalifragilisticexpialidocious";
#endif
    const char *s1, *s2; 

    if (argc == 3) {
        s1 = strdup(argv[1]);
        s2 = strdup(argv[2]);
    }
    else if (argc == 1) {
        /* read 2 strings from the standard input */
        if (scanf("%ms %ms", &s1, &s2) != 2) {
            fprintf(stderr, "invalid input\n");
            exit(1);
        }
    }
    else {
        fprintf(stderr, "usage: %s [<first string> <second string>]\n", argv[0]);
        exit(1);
    }

    int n1 = strlen(s1)+1; // null byte at the end is also counted
    int n2 = strlen(s2)+1;
           
    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_s1, n1);
    cudaMalloc((void**)&dev_s2, n2);
    cudaMalloc((void**)&dev_result, sizeof(int));
    
    cudaMemcpy(dev_s1, s1, n1, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s2, s2, n2, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = BLOCK_DIM;
    int numOfBlocks = 1;
 
    my_strcmp<<<numOfBlocks, threadsPerBlock>>>(dev_s1, n1, dev_s2, n2, dev_result);
 
    // copy the result back from the GPU to the CPU
    int result;
    cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("result is %d\n", result);
		
	    
    // free memory on the GPU side
    cudaFree(dev_s1);
    cudaFree(dev_s2);
    cudaFree(dev_result);
}

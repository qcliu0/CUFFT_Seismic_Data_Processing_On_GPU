#include<stdio.h>
#include<stdlib.h>

_global_ void testkernel(int *d_A,size_t size)
{
	int dx=blockDim.x*blockIdx.x+threadIdx.x;
	int dy=blockDim.y*blockIdx.y+threadIdx.y;
	
	if(blockIdx.x==0 && blockIdx.y==0)
		d_A[dx*size+dy]+=1;
	if(blockIdx.x==0 && blockIdx.y==1)
		d_A[dx*size+dy]+=2;
	if(blockIdx.x==1 && blockIdx.y==0)
		d_A[dx*size+dy]+=3;
	if(blockIdx.x==1 && blockIdx.y==1)
		d_A[dx*size+dy]+=4;
}

int main(int argc,char** argv)

{

	int h_A[8][8]={{1,1,1,1,2,2,2,2},
			{1,1,1,1,2,2,2,2},
			{1,1,1,1,2,2,2,2},
			{1,1,1,1,2,2,2,2},
			{3,3,3,3,4,4,4,4},
			{3,3,3,3,4,4,4,4},
			{3,3,3,3,4,4,4,4},
			{3,3,3,3,4,4,4,4}};

	int *d_A,*h_B;
	size_t size=8*8*sizeof(int);
	size_t rsize=8;
	dim3 dimgrid(2,2);
	dim3 dimblock(4,4);
	h_B=(int*)malloc(size);
	
	cudaMalloc((void**)&d_A,size);
	cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);

	testkernel<<<dimgrid,dimblock>>>(d_A,rsize);

	cudaMemcpy(h_B,d_A,size,cudaMemcpyDeviceToHost);

	for(int i=0;i<8;i++){
		for(int j=0;j<8;j++){
			printf("%2d",h_B[i*rsize+j]);	
		}printf("\n");
	}
	cudaFree(d_A);
	free(h_B);
}

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<cuda.h>
#include<cufft.h>
#include<cuda_runtime.h>
//#include <cutil_inline.h>
//#include <cutil.h>
int main()
{
	int nx,nt,i,ix,it;
	int NX,BATCH;
	float  **a_input;
	float  *input;
	float *amp;
	cufftHandle plan;
	cufftComplex *data;
	time_t t_z,t_f;
	nx=5300;
	nt=12001;
	a_input=(float**)calloc(nt,sizeof(float*));
	for(it=0;it<nt;it++){
		a_input[it]=(float*)calloc(nx,sizeof(float));
	}
	input=(float*)calloc(nx*nt,sizeof(float));
	amp=(float*)calloc(nt/2,sizeof(float));
	FILE *fp;
	fp=fopen("rec_u_3200.bin","rb");
	for(it=0;it<nt;it++){
		for(ix=0;ix<nx;ix++){
			fread(&a_input[it][ix],sizeof(float),1,fp);
		}
	}
	fclose(fp);

	for(ix=0;ix<nx;ix++){
		for(it=0;it<nt;it++){
			input[ix*nt+it]=a_input[it][ix];
		}
	}
	printf("re_transpose_done !!!\n");
	NX=nt;
	BATCH=10;
	cudaMalloc((void**)&data, sizeof(cufftComplex)*(NX/2+1)*BATCH);
	cudaMemcpy(data,input,NX*BATCH*sizeof(float),cudaMemcpyHostToDevice);

	t_z=time(NULL);	
	cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH);
	cufftExecR2C(plan, (cufftReal*)data, data);
	cufftDestroy(plan);
	t_f=time(NULL);
	printf("\nCalculating time:%f (s) \n\n",t_f-t_z);
	cudaMemcpy(input,data,nx*nt*sizeof(float),cudaMemcpyDeviceToHost);
	fp=fopen("bofore_cufft.bin","wb");
	for(it=0;it<nt;it++){
		for(ix=0;ix<1;ix++){
			fwrite(&a_input[it][ix],sizeof(float),1,fp);
		}
	}
	fclose(fp);
	fp=fopen("after_cufft.bin","wb");
	for(it=0;it<nt;it++){
		fwrite(&input[it],sizeof(float),1,fp);
	}
	fclose(fp);	
	cudaFree(data);
	for(i=0;i<nt/2;i++){
		amp[i]=sqrt(input[nt+2*i]*input[nt+2*i]+input[nt+2*i+1]*input[nt+2*i+1]);
	}
	fp=fopen("amp.bin","wb");
	fwrite(amp,sizeof(float),nt/2,fp);
	fclose(fp);
	return 0;

}

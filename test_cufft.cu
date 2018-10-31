#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<time.h>
#include<cuda.h>
#include<cufft.h>
#include<cuda_runtime.h>

#include"pfafft.c"//即使在无法调用su情况下，保证npfar有效
//#include <cutil_inline.h>
//#include <cutil.h>
int main()
{
	int nx,nt,nw,i,ix,it;//nx:the number of traces	nt:the number of time samples	
	int NX,BATCH,NX_t;
	float  **a_input;
	float  *input;
	float *amp;//testing
	cufftHandle plan;
	cufftComplex *data;
	clock_t t_z,t_f;
	nx=5300;
	nt=5001;
	NX=npfar(nt);//npfar,from seismic unix
	BATCH=nx;
	nw=NX/2+1;
	NX_t=(NX/2+1)*2;//NX_t makes sure that the array can meet the principle of "in-place"
	input=(float*)calloc(NX_t*BATCH,sizeof(float));
	amp=(float*)calloc(nw,sizeof(float));//testing
	a_input=(float**)calloc(nt,sizeof(float*));
	for(it=0;it<nt;it++){
		a_input[it]=(float*)calloc(nx,sizeof(float));
	}

	FILE *fp;
	fp=fopen("rec_u_3200.bin","rb");
	for(it=0;it<nt;it++){
		for(ix=0;ix<nx;ix++){
			fread(&a_input[it][ix],sizeof(float),1,fp);
		}
	}
	fclose(fp);

	for(ix=0;ix<BATCH;ix++){
		for(it=0;it<nt;it++){
			input[ix*NX_t+it]=a_input[it][ix];
		}
	}
	printf("re_transpose_done !!!\n");

	cudaMalloc((void**)&data, sizeof(cufftComplex)*(NX/2+1)*BATCH);
	cudaMemcpy(data,input,NX_t*BATCH*sizeof(float),cudaMemcpyHostToDevice);

	t_z=clock();	
	cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH);

	cufftExecR2C(plan, (cufftReal*)data, data);
	t_f=clock();
	cufftDestroy(plan);

	printf("\nCalculating time:%ld (cycles) \n\n",t_f-t_z);
	printf("\nCalculating time:%f (ms) \n\n",(double)(t_f-t_z)/CLOCKS_PER_SEC);

	cudaMemcpy(input,data,nx*nt*sizeof(float),cudaMemcpyDeviceToHost);
	fp=fopen("bofore_cufft.bin","wb");//testing
	for(it=0;it<nt;it++){
		for(ix=0;ix<1;ix++){
			fwrite(&a_input[it][ix],sizeof(float),1,fp);
		}
	}
	fclose(fp);
	fp=fopen("after_cufft.bin","wb");//testing
	for(it=0;it<nt;it++){
		fwrite(&input[it],sizeof(float),1,fp);
	}
	fclose(fp);	
	cudaFree(data);
	for(i=0;i<nt/2;i++){
		amp[i]=sqrt(input[499*NX_t+2*i]*input[499*NX_t+2*i]+input[499*NX_t+2*i+1]*input[499*NX_t+2*i+1]);
	}
	fp=fopen("amp.bin","wb");//testing
	fwrite(amp,sizeof(float),nt/2,fp);
	fclose(fp);
	return 0;

}

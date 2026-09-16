#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s: %s\n",#x,cudaGetErrorString(e));return 1;}}while(0)
int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; CK(cudaSetDevice(dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
  printf("dev %d %s pci %02x:%02x\n",dev,p.name,p.pciBusID,p.pciDeviceID);
  size_t n=256ull<<20; void *h,*d,*d2; CK(cudaMallocHost(&h,n)); CK(cudaMalloc(&d,n)); CK(cudaMalloc(&d2,n));
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  for(int i=0;i<3;i++) CK(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice));
  cudaEventRecord(a); for(int i=0;i<10;i++) cudaMemcpyAsync(d,h,n,cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("H2D pinned : %.2f GB/s\n",10.0*n/ms/1e6);
  cudaEventRecord(a); for(int i=0;i<10;i++) cudaMemcpyAsync(h,d,n,cudaMemcpyDeviceToHost); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("D2H pinned : %.2f GB/s\n",10.0*n/ms/1e6);
  cudaEventRecord(a); for(int i=0;i<10;i++) cudaMemcpyAsync(d2,d,n,cudaMemcpyDeviceToDevice); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("D2D local  : %.2f GB/s\n",10.0*n/ms/1e6);
  // small-transfer latency: 8 KiB activations
  size_t s=8192; cudaEventRecord(a); for(int i=0;i<1000;i++) cudaMemcpyAsync(d,h,s,cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("H2D 8KiB   : %.1f us each\n",ms*1000/1000);
  cudaEventRecord(a); for(int i=0;i<1000;i++) cudaMemcpyAsync(h,d,s,cudaMemcpyDeviceToHost); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b);
  printf("D2H 8KiB   : %.1f us each\n",ms*1000/1000);
  return 0;}

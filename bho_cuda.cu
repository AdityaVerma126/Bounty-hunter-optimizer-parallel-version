/* bho_cuda.cu : CUDA version. One thread block = one independent run; the whole
 * optimisation loop runs inside ONE kernel (no per-iteration host<->device traffic).
 * Threads of the block loop over hunters; thread 0 does the tiny serial phases.
 * The O(N^2*D) Explorpolis check is spread over (i,j) pairs so all threads work.
 * Build: nvcc -O3 -arch=sm_XX --fmad=false -o bho_cuda bho_cuda.cu   (XX = your GPU, e.g. 75/86/89) */
#include <cuda_runtime.h>
#include "bho_core.h"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)

__global__ void bho_kernel(BHOParams pp, double *dd, int *di, const double *shift, double *curves){
  int run=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
  BHOParams p=pp; p.seed=pp.seed+(uint64_t)run*7919ULL;            /* same seeding as CPU versions */
  BHOState s; bho_bind(&s,&p,dd+(size_t)run*bho_dbl_count(&p),di+(size_t)run*bho_int_count(&p));
  s.shift=shift;
  double *curve=curves+(size_t)run*(p.kmax+1);
  int N=p.N;
  if(tid==0) curve[0]=s.PF[s.ig[0]];
  __syncthreads();
  for(int k=1;k<=p.kmax;k++){
    if(tid==0) bho_phase_stats(&p,&s,k);
    __syncthreads();
    for(int i=tid;i<N;i+=nt) bho_phase_rank_ref(&p,&s,k,i);
    __syncthreads();
    if(tid==0) bho_phase_reassign(&p,&s,k);
    __syncthreads();
    for(int i=tid;i<N;i+=nt) bho_phase_candidate(&p,&s,k,i);
    __syncthreads();
    /* Explorpolis: start with "accept", then any (i,j) pair that blocks candidate i clears the flag */
    for(int i=tid;i<N;i+=nt) s.acc[i]=1;
    __syncthreads();
    for(int pr=tid;pr<N*N;pr+=nt){
      int i=pr/N, j=pr%N;
      if(s.Ft[i]<s.F[i] || s.acc[i]==0) continue;                  /* already better than parent / already blocked */
      if(bho_pair_blocks(&p,&s,i,j)) s.acc[i]=0;                   /* benign race: all writers store 0 */
    }
    __syncthreads();
    for(int i=tid;i<N;i+=nt) bho_phase_apply(&p,&s,i);
    __syncthreads();
    if(tid==0) bho_phase_finish(&p,&s,k,curve);
    __syncthreads();
  }
}

int main(int argc, char **argv){
  Cfg cfg=cfg_default(); cfg_parse(&cfg,argc,argv);
  const char *out=cfg.out?cfg.out:"results_cuda.csv"; FILE *fo=fopen(out,"w");
  fprintf(fo,"fid,name,best_err,mean_err,std_err,t_init,t_opt\n");
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0)); CK(cudaFree(0));   /* context warm-up */
  printf("BHO CUDA on %s | D=%d runs=%d\n",prop.name,cfg.D,cfg.runs);
  int threads=256;
  for(int fi=0;fi<NFUNC;fi++){
    int fid=BENCH_IDS[fi]; if(cfg.fid>0 && cfg.fid!=fid) continue;
    BHOParams p=bho_params(&cfg,fid);
    size_t dn=bho_dbl_count(&p), in=bho_int_count(&p);
    double *shift=(double*)malloc(cfg.D*sizeof(double)); bench_make_shift(fid,cfg.D,shift);
    double *hd=(double*)calloc(dn*cfg.runs,sizeof(double)); int *hi=(int*)calloc(in*cfg.runs,sizeof(int));
    /* --- host-side initialisation (k-means etc.) --- */
    double t0=now_sec();
    for(int r=0;r<cfg.runs;r++){
      BHOParams pr=p; pr.seed=cfg.seed+(uint64_t)r*7919ULL;
      BHOState s; bho_bind(&s,&pr,hd+dn*r,hi+in*r); s.shift=shift;
      bho_init(&pr,&s,0);
    }
    double t1=now_sec();
    p.seed=cfg.seed;                                              /* kernel adds run*7919 */
    double *d_d,*d_sh,*d_c; int *d_i; size_t cn=(size_t)cfg.runs*(p.kmax+1);
    CK(cudaMalloc(&d_d,dn*cfg.runs*sizeof(double))); CK(cudaMalloc(&d_i,in*cfg.runs*sizeof(int)));
    CK(cudaMalloc(&d_sh,cfg.D*sizeof(double)));      CK(cudaMalloc(&d_c,cn*sizeof(double)));
    CK(cudaMemcpy(d_d,hd,dn*cfg.runs*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_i,hi,in*cfg.runs*sizeof(int),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sh,shift,cfg.D*sizeof(double),cudaMemcpyHostToDevice));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    bho_kernel<<<cfg.runs,threads>>>(p,d_d,d_i,d_sh,d_c);
    CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    double *hc=(double*)malloc(cn*sizeof(double));
    CK(cudaMemcpy(hc,d_c,cn*sizeof(double),cudaMemcpyDeviceToHost));
    double *err=(double*)malloc(cfg.runs*sizeof(double));
    for(int r=0;r<cfg.runs;r++) err[r]=hc[(size_t)r*(p.kmax+1)+p.kmax]-bench_bias(fid);
    double mn,mean,sd; stats(err,cfg.runs,&mn,&mean,&sd);
    printf("F%-2d %-10s best=%.4e mean=%.4e std=%.4e | init %.3fs opt %.3fs (kernel)\n",fid,bench_name(fid),mn,mean,sd,t1-t0,ms/1000.0);
    fprintf(fo,"%d,%s,%.10e,%.10e,%.10e,%.6f,%.6f\n",fid,bench_name(fid),mn,mean,sd,t1-t0,ms/1000.0);
    cudaFree(d_d);cudaFree(d_i);cudaFree(d_sh);cudaFree(d_c);
    free(hd);free(hi);free(shift);free(hc);free(err);
  }
  fclose(fo); return 0;
}

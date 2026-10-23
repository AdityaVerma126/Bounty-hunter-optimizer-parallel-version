/* bho_common.h : RNG, benchmark functions, k-means, CLI, timing.
 * Shared by the sequential, OpenMP and CUDA versions of BHO. */
#ifndef BHO_COMMON_H
#define BHO_COMMON_H
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#define BHO_PI 3.14159265358979323846

/* ---------- STEP 0a: stateless RNG ----------------------------------------
 * urand(seed,a,b,c) is a pure function (hash), so every random number is
 * identified by (run, iteration, individual, dimension). Sequential, OpenMP
 * and CUDA therefore draw the SAME numbers -> results are comparable and
 * independent of thread scheduling. */
HD static inline uint64_t mix64(uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
HD static inline double urand(uint64_t seed, uint64_t a, uint64_t b, uint64_t c){
  uint64_t h = mix64(seed);
  h = mix64(h ^ a);
  h = mix64(h ^ (b * 0x9E3779B97F4A7C15ULL));
  h = mix64(h ^ (c * 0xC2B2AE3D27D4EB4FULL));
  return (double)(h >> 11) * (1.0 / 9007199254740992.0);   /* [0,1) */
}

/* ---------- STEP 0b: benchmark functions -----------------------------------
 * Shifted (NOT rotated) versions of CEC2017 F1,F3,F4,F5,F9,F10, search range
 * [-100,100], bias = 100*fid. The shift vector is generated pseudo-randomly,
 * so these are NOT numerically identical to the official CEC2017 files.
 * For official values compile with -DUSE_OFFICIAL_CEC17 (host only, see notes). */
#define NFUNC 6
static const int BENCH_IDS[NFUNC] = {1, 3, 4, 5, 9, 10};
static const char *bench_name(int fid){
  switch(fid){case 1:return "BentCigar"; case 3:return "Zakharov"; case 4:return "Rosenbrock";
              case 5:return "Rastrigin"; case 9:return "Levy"; case 10:return "Schwefel";}
  return "?";
}
static inline double bench_bias(int fid){ return 100.0 * fid; }
static void bench_make_shift(int fid, int D, double *sh){
  for(int j = 0; j < D; j++) sh[j] = -80.0 + 160.0 * urand(0x5EEDULL, (uint64_t)fid, 0, (uint64_t)j);
}
HD static inline double bench_eval(int fid, const double *x, const double *sh, int D){
  double f = 0.0; int j;
  switch(fid){
  case 1: { double z0 = x[0]-sh[0]; f = z0*z0;
            for(j=1;j<D;j++){ double z=x[j]-sh[j]; f += 1e6*z*z; } break; }
  case 3: { double s1=0.0,s2=0.0;
            for(j=0;j<D;j++){ double z=x[j]-sh[j]; s1+=z*z; s2+=0.5*(j+1)*z; }
            f = s1 + s2*s2 + s2*s2*s2*s2; break; }
  case 4: { for(j=0;j<D-1;j++){
              double z=0.02048*(x[j]-sh[j])+1.0, z1=0.02048*(x[j+1]-sh[j+1])+1.0, t=z*z-z1;
              f += 100.0*t*t + (z-1.0)*(z-1.0); } break; }
  case 5: { for(j=0;j<D;j++){ double z=0.0512*(x[j]-sh[j]);
              f += z*z - 10.0*cos(2.0*BHO_PI*z) + 10.0; } break; }
  case 9: { double w0 = 1.0+0.025*(x[0]-sh[0]); double s0 = sin(BHO_PI*w0); f = s0*s0;
            for(j=0;j<D-1;j++){ double w=1.0+0.025*(x[j]-sh[j]); double s=sin(BHO_PI*w+1.0);
              f += (w-1.0)*(w-1.0)*(1.0+10.0*s*s); }
            double wd=1.0+0.025*(x[D-1]-sh[D-1]); double sd=sin(2.0*BHO_PI*wd);
            f += (wd-1.0)*(wd-1.0)*(1.0+sd*sd); break; }
  case 10:{ double s=0.0;
            for(j=0;j<D;j++){
              double z=10.0*(x[j]-sh[j])+420.9687462275036, g;
              if(z>500.0){ double t=500.0-fmod(z,500.0); g=t*sin(sqrt(fabs(t)))-(z-500.0)*(z-500.0)/(10000.0*D); }
              else if(z<-500.0){ double t=fmod(fabs(z),500.0)-500.0; g=t*sin(sqrt(fabs(t)))-(z+500.0)*(z+500.0)/(10000.0*D); }
              else g=z*sin(sqrt(fabs(z)));
              s+=g; }
            f = 418.9829*D - s; break; }
  default:{ for(j=0;j<D;j++){ double z=x[j]-sh[j]; f+=z*z; } }
  }
  return f + 100.0*fid;
}

/* ---------- STEP 0c: k-means (host only) -----------------------------------
 * Used twice: (1) 10N oversampled points -> N centroids (initial population),
 *             (2) N hunters -> Ng groups.  `par` enables OpenMP on the assign step. */
static void kmeans(const double *P, int n, int D, int K, int iters, uint64_t seed,
                   double *C, int *asg, int par){
  int *perm = (int*)malloc((size_t)n*sizeof(int));
  double *sum = (double*)calloc((size_t)K*D, sizeof(double));
  int *cnt = (int*)calloc((size_t)K, sizeof(int));
  for(int i=0;i<n;i++) perm[i]=i;
  for(int k=0;k<K;k++){                           /* K distinct random seeds */
    int r = k + (int)(urand(seed,0,(uint64_t)k,0)*(n-k)); if(r>=n) r=n-1;
    int t=perm[k]; perm[k]=perm[r]; perm[r]=t;
    memcpy(C+(size_t)k*D, P+(size_t)perm[k]*D, (size_t)D*sizeof(double));
  }
  for(int it=0; it<=iters; it++){
    #pragma omp parallel for if(par) schedule(static)
    for(int i=0;i<n;i++){
      int best=0; double bd=1e300;
      for(int k=0;k<K;k++){
        double d=0.0; const double *c=C+(size_t)k*D, *p=P+(size_t)i*D;
        for(int j=0;j<D;j++){ double t=p[j]-c[j]; d+=t*t; }
        if(d<bd){bd=d;best=k;}
      }
      asg[i]=best;
    }
    if(it==iters) break;
    memset(sum,0,(size_t)K*D*sizeof(double)); memset(cnt,0,(size_t)K*sizeof(int));
    for(int i=0;i<n;i++){ int a=asg[i]; cnt[a]++; for(int j=0;j<D;j++) sum[(size_t)a*D+j]+=P[(size_t)i*D+j]; }
    for(int k=0;k<K;k++){
      if(cnt[k]>0){ for(int j=0;j<D;j++) C[(size_t)k*D+j]=sum[(size_t)k*D+j]/cnt[k]; }
      else { int r=(int)(urand(seed,1+(uint64_t)it,(uint64_t)k,1)*n); if(r>=n) r=n-1;
             memcpy(C+(size_t)k*D, P+(size_t)r*D, (size_t)D*sizeof(double)); }
    }
  }
  free(perm); free(sum); free(cnt);
}

/* ---------- CLI + helpers ---------------------------------------------------*/
typedef struct { int D, runs, fid, threads, runmode, Ng, MPPC; uint64_t seed;
                 double rho, tau, rh0; const char *out; } Cfg;
static Cfg cfg_default(void){
  Cfg c; c.D=30; c.runs=10; c.fid=-1; c.threads=0; c.runmode=0; c.Ng=4; c.MPPC=2;
  c.seed=1; c.rho=0.25; c.tau=0.3; c.rh0=0.3; c.out=NULL; return c; }
static void cfg_parse(Cfg *c, int argc, char **argv){
  for(int a=1;a<argc;a++){
    #define OPT(name) (!strcmp(argv[a],name) && a+1<argc)
    if(OPT("-D")) c->D=atoi(argv[++a]);
    else if(OPT("-runs")) c->runs=atoi(argv[++a]);
    else if(OPT("-f")) c->fid=atoi(argv[++a]);
    else if(OPT("-seed")) c->seed=(uint64_t)atoll(argv[++a]);
    else if(OPT("-t")) c->threads=atoi(argv[++a]);
    else if(OPT("-mode")) c->runmode=!strcmp(argv[++a],"runs");   /* omp: inner | runs */
    else if(OPT("-ng")) c->Ng=atoi(argv[++a]);
    else if(OPT("-rho")) c->rho=atof(argv[++a]);
    else if(OPT("-tau")) c->tau=atof(argv[++a]);
    else if(OPT("-rh")) c->rh0=atof(argv[++a]);
    else if(OPT("-o")) c->out=argv[++a];
  }
}
static double now_sec(void){
#ifdef _WIN32
  LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
  return (double)t.QuadPart/(double)f.QuadPart;
#else
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec;
#endif
}
static void stats(const double *v, int n, double *mn, double *mean, double *sd){
  double s=0,m=1e300; for(int i=0;i<n;i++){ s+=v[i]; if(v[i]<m) m=v[i]; }
  double mu=s/n, q=0; for(int i=0;i<n;i++) q+=(v[i]-mu)*(v[i]-mu);
  *mn=m; *mean=mu; *sd=sqrt(q/(n>1?n-1:1));
}
#endif

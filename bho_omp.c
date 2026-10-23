/* bho_omp.c : OpenMP version.  Two ways to use the cores (-mode):
 *   inner : runs executed one after another, the loops over hunters are parallel
 *   runs  : independent runs executed in parallel (each run sequential inside)  <- usually scales best
 * Build: gcc -O3 -march=native -ffp-contract=off -fopenmp -o bho_omp bho_omp.c -lm */
#include <omp.h>
#include "bho_core.h"

static void bho_run_loop(const BHOParams *p, BHOState *s, double *curve, int par){
  int N=p->N, k;
  curve[0]=s->PF[s->ig[0]];
  for(k=1;k<=p->kmax;k++){
    bho_phase_stats(p,s,k);                                   /* serial, O(N*Ng) */
    #pragma omp parallel for if(par) schedule(static)
    for(int i=0;i<N;i++) bho_phase_rank_ref(p,s,k,i);
    bho_phase_reassign(p,s,k);                                /* serial, tiny */
    #pragma omp parallel for if(par) schedule(static)
    for(int i=0;i<N;i++) bho_phase_candidate(p,s,k,i);        /* O(N*D) + objective */
    #pragma omp parallel for if(par) schedule(static)
    for(int i=0;i<N;i++) bho_phase_select(p,s,i);             /* O(N^2*D): dominant cost */
    #pragma omp parallel for if(par) schedule(static)
    for(int i=0;i<N;i++) bho_phase_apply(p,s,i);
    bho_phase_finish(p,s,k,curve);                            /* serial */
  }
}

int main(int argc, char **argv){
  Cfg cfg=cfg_default(); cfg_parse(&cfg,argc,argv);
  if(cfg.threads>0) omp_set_num_threads(cfg.threads);
  int runs_mode=cfg.runmode, inner=!runs_mode;
  const char *out=cfg.out?cfg.out:"results_omp.csv"; FILE *fo=fopen(out,"w");
  fprintf(fo,"fid,name,best_err,mean_err,std_err,t_init,t_opt\n");
  printf("BHO OpenMP | D=%d runs=%d threads=%d mode=%s\n",cfg.D,cfg.runs,omp_get_max_threads(),runs_mode?"runs":"inner");
  for(int fi=0;fi<NFUNC;fi++){
    int fid=BENCH_IDS[fi]; if(cfg.fid>0 && cfg.fid!=fid) continue;
    BHOParams p=bho_params(&cfg,fid);
    double *shift=(double*)malloc(cfg.D*sizeof(double)); bench_make_shift(fid,cfg.D,shift);
    BHOState *S=(BHOState*)malloc(cfg.runs*sizeof(BHOState));
    BHOParams *P=(BHOParams*)malloc(cfg.runs*sizeof(BHOParams));
    double *curves=(double*)malloc((size_t)cfg.runs*(p.kmax+1)*sizeof(double));
    double *err=(double*)malloc(cfg.runs*sizeof(double));
    double t0=omp_get_wtime();
    #pragma omp parallel for if(runs_mode) schedule(dynamic)
    for(int r=0;r<cfg.runs;r++){ P[r]=p; P[r].seed=cfg.seed+(uint64_t)r*7919ULL;
      bho_alloc(&P[r],&S[r],shift); bho_init(&P[r],&S[r],inner); }
    double t1=omp_get_wtime();
    #pragma omp parallel for if(runs_mode) schedule(dynamic)
    for(int r=0;r<cfg.runs;r++) bho_run_loop(&P[r],&S[r],curves+(size_t)r*(p.kmax+1),inner);
    double t2=omp_get_wtime();
    for(int r=0;r<cfg.runs;r++){ err[r]=S[r].PF[S[r].ig[0]]-bench_bias(fid); bho_free(&S[r]); }
    double mn,mean,sd; stats(err,cfg.runs,&mn,&mean,&sd);
    printf("F%-2d %-10s best=%.4e mean=%.4e std=%.4e | init %.3fs opt %.3fs\n",fid,bench_name(fid),mn,mean,sd,t1-t0,t2-t1);
    fprintf(fo,"%d,%s,%.10e,%.10e,%.10e,%.6f,%.6f\n",fid,bench_name(fid),mn,mean,sd,t1-t0,t2-t1);
    free(S);free(P);free(curves);free(err);free(shift);
  }
  fclose(fo); return 0;
}

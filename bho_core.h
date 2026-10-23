/* bho_core.h : the Bounty Hunter Optimizer, written as small per-individual
 * "phase" functions that compile for host AND device (HD).
 *   sequential : loops over i
 *   OpenMP     : same loops + "#pragma omp parallel for"
 *   CUDA       : one thread block per independent run, threads loop over i
 * Minimisation is assumed (the paper's pseudo-code maximises "clue value"). */
#ifndef BHO_CORE_H
#define BHO_CORE_H
#include "bho_common.h"

#define KM_ITERS 5
#ifndef ROUGH_VARIANT
#define ROUGH_VARIANT 0   /* 0 = literal paper Eq.9 (default); 1,2 = alternatives, see notes */
#endif
enum { T_INIT=11, T_CF=12, T_THOR=13, T_ROUGH=14, T_REAS=15, T_QPRS=16, T_SIGN=17 };
#define KEY(tag,k) ((uint64_t)(tag) | ((uint64_t)(k) << 8))

typedef struct {
  int N, D, Ng, kmax, fid, MPPC, guardMAX;
  double rho, tau, rh0, lo, hi;
  uint64_t seed;
} BHOParams;

typedef struct {            /* pointers into one run's workspace */
  double *X,*Xt,*PX;        /* positions, test positions, personal-best positions  [N*D] */
  double *F,*Ft,*PF;        /* fitness, test fitness, personal-best fitness        [N]   */
  double *cc,*mm,*Rr,*sco;  /* contribution c_i (16), step m_i (18), rank ratio, score (5) */
  double *cf;               /* reference factors c1,c2,c3 per hunter               [3N]  */
  double *Cg,*qa,*qb;       /* group contribution (15), qubit amplitudes alpha,beta [Ng] */
  double *sc;               /* sc[0]=r_h, sc[1]=sum F, sc[2]=positivity offset            */
  int *grp,*poor,*mode,*acc;/* group id, poor-count P_i, mode (0 rough,1 thorough,2 reassigned), accept flag */
  int *gcnt,*gbst,*gmem,*ig;/* group sizes, group-best index, member lists [Ng*N], ig[0]=global best */
  const double *shift;
  double *_d; int *_i;      /* owned buffers (host only) */
} BHOState;

HD static inline size_t bho_dbl_count(const BHOParams *p){
  return (size_t)3*p->N*p->D + 10*(size_t)p->N + 3*(size_t)p->Ng + 3; }
HD static inline size_t bho_int_count(const BHOParams *p){
  return 4*(size_t)p->N + 2*(size_t)p->Ng + (size_t)p->Ng*p->N + 1; }

HD static void bho_bind(BHOState *s, const BHOParams *p, double *d, int *v){
  size_t N=p->N, ND=(size_t)p->N*p->D, Ng=p->Ng, o=0;
  s->X=d+o; o+=ND; s->Xt=d+o; o+=ND; s->PX=d+o; o+=ND;
  s->F=d+o; o+=N; s->Ft=d+o; o+=N; s->PF=d+o; o+=N; s->cc=d+o; o+=N;
  s->mm=d+o; o+=N; s->Rr=d+o; o+=N; s->sco=d+o; o+=N; s->cf=d+o; o+=3*N;
  s->Cg=d+o; o+=Ng; s->qa=d+o; o+=Ng; s->qb=d+o; o+=Ng; s->sc=d+o;
  o=0;
  s->grp=v+o; o+=N; s->poor=v+o; o+=N; s->mode=v+o; o+=N; s->acc=v+o; o+=N;
  s->gcnt=v+o; o+=Ng; s->gbst=v+o; o+=Ng; s->gmem=v+o; o+=Ng*N; s->ig=v+o;
  s->_d=NULL; s->_i=NULL;
}
static void bho_alloc(const BHOParams *p, BHOState *s, const double *shift){
  double *d=(double*)calloc(bho_dbl_count(p),sizeof(double));
  int *v=(int*)calloc(bho_int_count(p),sizeof(int));
  bho_bind(s,p,d,v); s->shift=shift; s->_d=d; s->_i=v;
}
static void bho_free(BHOState *s){ free(s->_d); free(s->_i); }

static BHOParams bho_params(const Cfg *c, int fid){
  BHOParams p; p.D=c->D; p.N=c->D;                 /* N = 1*D (paper, Sec. 3)   */
  p.Ng=c->Ng; p.fid=fid; p.MPPC=c->MPPC; p.guardMAX=100;
  p.kmax = 500*p.D/p.N - 1;                        /* 500*D FEs, N FEs per iter */
  p.rho=c->rho; p.tau=c->tau; p.rh0=c->rh0; p.lo=-100.0; p.hi=100.0; p.seed=c->seed;
  return p;
}

#ifdef USE_OFFICIAL_CEC17            /* host only; official code is not thread-safe */
extern void cec17_test_func(double*, double*, int, int, int);
static inline double bho_eval(int fid,const double*x,const double*sh,int D){
  double f; (void)sh; cec17_test_func((double*)x,&f,D,1,fid); return f; }
#else
#define bho_eval bench_eval
#endif

HD static inline double clampd(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

/* Group bookkeeping: member lists, group-best (by personal best), global best. */
HD static void bho_build_groups(const BHOParams *p, BHOState *s){
  int N=p->N, Ng=p->Ng, i, l;
  for(l=0;l<Ng;l++){ s->gcnt[l]=0; s->gbst[l]=-1; }
  for(i=0;i<N;i++){
    int g=s->grp[i]; s->gmem[(size_t)g*N + s->gcnt[g]++] = i;
    if(s->gbst[g]<0 || s->PF[i]<s->PF[s->gbst[g]]) s->gbst[g]=i;
  }
  int b=0; for(i=1;i<N;i++) if(s->PF[i]<s->PF[b]) b=i;
  s->ig[0]=b;
}

/* ===== STEP 1 (host): initialisation (Sec. 2.1.1-2.1.2) ===================== */
static void bho_init(const BHOParams *p, BHOState *s, int par){
  int N=p->N, D=p->D, Ng=p->Ng, No=10*N, i, j, l;
  double *Xo=(double*)malloc((size_t)No*D*sizeof(double));
  int *asg=(int*)malloc((size_t)No*sizeof(int));
  for(i=0;i<No;i++) for(j=0;j<D;j++)                     /* 10N oversampled clues (Eq.1) */
    Xo[(size_t)i*D+j]=p->lo+(p->hi-p->lo)*urand(p->seed,T_INIT,(uint64_t)i,(uint64_t)j);
  kmeans(Xo,No,D,N,KM_ITERS,p->seed^0xA5A5ULL,s->X,asg,par);   /* N centroids = population (Eq.2) */
  free(Xo); free(asg);
  asg=(int*)malloc((size_t)N*sizeof(int));
  double *gc=(double*)malloc((size_t)Ng*D*sizeof(double));
  kmeans(s->X,N,D,Ng,KM_ITERS,p->seed^0x5A5AULL,gc,asg,par);   /* Ng pursuit groups */
  for(i=0;i<N;i++){
    s->grp[i]=asg[i]; s->poor[i]=0; s->mode[i]=1; s->acc[i]=0;
    for(j=0;j<D;j++) s->PX[(size_t)i*D+j]=s->X[(size_t)i*D+j];
    s->F[i]=bho_eval(p->fid,s->X+(size_t)i*D,s->shift,D); s->PF[i]=s->F[i];
    for(l=0;l<3;l++) s->cf[3*i+l]=urand(p->seed,T_CF,(uint64_t)i,(uint64_t)l);  /* c1..c3 in [0,1] */
  }
  for(l=0;l<Ng;l++){ s->Cg[l]=0.0; s->qa[l]=0.0; s->qb[l]=0.0; }
  s->sc[0]=p->rh0; s->sc[1]=1.0; s->sc[2]=0.0;
  free(asg); free(gc);
  bho_build_groups(p,s);
}

/* ===== STEP 2 (serial, O(N*Ng)): contributions, group scores (Eq.15,16,14) === */
HD static void bho_phase_stats(const BHOParams *p, BHOState *s, int k){
  int N=p->N, Ng=p->Ng, i, l;
  bho_build_groups(p,s);
  double mn=s->F[0]; for(i=1;i<N;i++) if(s->F[i]<mn) mn=s->F[i];
  double off = (mn>0.0)?0.0:(1.0-mn);          /* Eq.16/18 assume F>0; shift if not */
  double sum=0.0; for(i=0;i<N;i++) sum+=s->F[i]+off;
  s->sc[1]=sum; s->sc[2]=off;
  for(i=0;i<N;i++) s->cc[i]=1.0-(s->F[i]+off)/sum;                 /* Eq.16 */
  for(l=0;l<Ng;l++){                                               /* Eq.15 */
    double add=0.0; for(i=0;i<N;i++) if(s->grp[i]==l) add+=s->cc[i];
    s->Cg[l]=(1.0-p->rho)*s->Cg[l]+add;
  }
  if(k==1){                                                        /* Eq.14: alpha_l = C_l / sum C */
    double sc=0.0; for(l=0;l<Ng;l++) sc+=s->Cg[l];
    for(l=0;l<Ng;l++){ double a=s->Cg[l]/sc; s->qa[l]=a; s->qb[l]=sqrt(fmax(0.0,1.0-a*a)); }
  }
}

/* ===== STEP 3 (parallel over i): rank ratio, step m_i, reference factors ===== */
HD static void bho_phase_rank_ref(const BHOParams *p, BHOState *s, int k, int i){
  int N=p->N, g=s->grp[i], j, t, rank=1;
  for(j=0;j<N;j++)
    if(s->grp[j]==g && (s->F[j]<s->F[i] || (s->F[j]==s->F[i] && j<i))) rank++;
  double R=(double)rank/(double)s->gcnt[g]; s->Rr[i]=R;           /* R in (0,1], 1/size = best */
  s->mm[i]=1.0-((double)k/(2.0*p->kmax))*(s->F[i]+s->sc[2])/s->sc[1];   /* Eq.18 */
  double *c=s->cf+3*i, cbar=(c[0]+c[1]+c[2])/3.0;                 /* Eq.7 */
  if(R<=0.2 || R>=0.8){
    double d=R/10.0;
    for(t=0;t<3;t++){
      int up=(R<=0.2)?(c[t]>=cbar):(c[t]<cbar);
      c[t]= up? fmin(c[t]+d,0.8) : fmax(c[t]-d,0.2);
    }
  }
}

/* ----- QPRS: quantum rotation gate (Eq.17, Table 1) -------------------------- */
HD static void bho_qrotate(const BHOParams *p, BHOState *s, int k, int i, int sel){
  int Ng=p->Ng, l, best=0;
  for(l=1;l<Ng;l++) if(s->Cg[l]>s->Cg[best]) best=l;
  int Qb=(sel==best);
  for(l=0;l<Ng;l++){
    int Ql=(l==sel), cond=(s->Cg[l]>s->Cg[best]);
    int sPos=0,sNeg=0,sA0=0,sB0=0;            /* 2 means "+/- 0.01*pi, random" */
    if(Ql==0 && Qb==1){ if(!cond){sPos= 1;sNeg=-1;sA0=0;sB0=2;} else {sPos=-1;sNeg= 1;sA0=2;sB0=0;} }
    if(Ql==1 && Qb==0){ if(!cond){sPos=-1;sNeg= 1;sA0=2;sB0=0;} else {sPos= 1;sNeg=-1;sA0=0;sB0=2;} }
    double a=s->qa[l], b=s->qb[l]; int sg;
    if(a==0.0) sg=sA0; else if(b==0.0) sg=sB0; else sg=(a*b>0.0)?sPos:sNeg;
    if(sg==2) sg=(urand(p->seed,KEY(T_SIGN,k),(uint64_t)i,(uint64_t)l)<0.5)?-1:1;
    double th=sg*0.01*BHO_PI, ct=cos(th), st=sin(th);
    s->qa[l]=ct*a-st*b; s->qb[l]=st*a+ct*b;     /* keeps alpha^2+beta^2 = 1 */
  }
}
/* ----- QPRS measurement = Algorithm 1 ---------------------------------------- */
HD static int bho_qprs(const BHOParams *p, BHOState *s, int k, int i){
  int Ng=p->Ng, l, Ns=0, guard=0, phi[16];
  for(l=0;l<Ng;l++) phi[l]=0;
  uint64_t key=KEY(T_QPRS,k);
  while(Ns!=1 && guard<p->guardMAX){
    if(Ns==0){
      for(l=0;l<Ng;l++) if(phi[l]==0 && s->gcnt[l]>0){
        double r=urand(p->seed,key,(uint64_t)i,(uint64_t)guard*16+l);
        if(r>s->qa[l]*s->qa[l]){ phi[l]=1; Ns++; } }
    } else {
      for(l=0;l<Ng;l++) if(phi[l]==1){
        double r=urand(p->seed,key,(uint64_t)i,(uint64_t)guard*16+l);
        if(r<=s->qa[l]*s->qa[l]){ phi[l]=0; Ns--; } }
    }
    guard++;
  }
  int sel=-1;
  if(Ns==1){ for(l=0;l<Ng;l++) if(phi[l]) sel=l; }
  else {                                         /* guardMAX fallback: random non-empty group */
    int t=(int)(urand(p->seed,key,(uint64_t)i,9999)*Ng);
    for(int q=0;q<Ng;q++){ l=(t+q)%Ng; if(s->gcnt[l]>0){ sel=l; break; } }
  }
  bho_qrotate(p,s,k,i,sel);
  return sel;
}

/* ===== STEP 4 (serial, tiny): mode selection + hunter reassignment (Sec 2.1.4)= */
HD static void bho_phase_reassign(const BHOParams *p, BHOState *s, int k){
  int N=p->N, i, any=0;
  double rh=s->sc[0];
  for(i=0;i<N;i++){
    int m;
    if(s->Rr[i]>0.8){                      /* bottom 20 % */
      s->poor[i]++;
      if(s->poor[i]>=p->MPPC){ s->poor[i]=0; s->grp[i]=bho_qprs(p,s,k,i); any=1; m=2; }
      else m=(s->Rr[i]<=rh)?1:0;
    } else { s->poor[i]=0; m=(s->Rr[i]<=rh)?1:0; }   /* Eq.3 */
    s->mode[i]=m;
  }
  if(any) bho_build_groups(p,s);
}

/* ===== STEP 5 (parallel over i): generate candidate + evaluate =============== */
HD static void bho_phase_candidate(const BHOParams *p, BHOState *s, int k, int i){
  int N=p->N, D=p->D, j, g=s->grp[i];
  double *xt=s->Xt+(size_t)i*D; const double *x=s->X+(size_t)i*D;
  double m=s->mm[i]; const double *c=s->cf+3*i;
  int mo=s->mode[i];
  if(mo!=0){                                   /* thorough (Eq.6) or reassigned (Eq.12) */
    const double *gB=s->PX+(size_t)s->ig[0]*D;
    const double *tB=s->PX+(size_t)s->gbst[g]*D;
    const double *pB=s->PX+(size_t)i*D;
    if(mo==2){                                 /* "better" hunter of the new group */
      int cnt=s->gcnt[g], st=(int)(urand(p->seed,KEY(T_REAS,k),(uint64_t)i,0)*cnt), b=-1;
      for(int t=0;t<cnt;t++){ int mb=s->gmem[(size_t)g*N+(st+t)%cnt]; if(s->PF[mb]<s->PF[i]){ b=mb; break; } }
      if(b>=0) pB=s->PX+(size_t)b*D;
    }
    uint64_t key=KEY(T_THOR,k);
    for(j=0;j<D;j++){
      double r1=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+0);
      double r2=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+1);
      double r3=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+2);
      xt[j]=x[j]+m*(c[0]*r1*(gB[j]-x[j]) + c[1]*r2*(tB[j]-x[j]) + c[2]*r3*(pB[j]-x[j]));
    }
  } else {                                     /* rough search (Eq.8, 9) */
    uint64_t key=KEY(T_ROUGH,k);
    int cnt=s->gcnt[g], r;
    if(cnt>=2){ int t=(int)(urand(p->seed,key,(uint64_t)i,7)*(cnt-1));
                r=s->gmem[(size_t)g*N+t]; if(r==i) r=s->gmem[(size_t)g*N+cnt-1]; }
    else { r=(int)(urand(p->seed,key,(uint64_t)i,7)*(N-1)); if(r>=i) r++; }
    int toward=(s->F[r]<s->F[i]);              /* better reference -> move toward, else away */
    const double *xr=s->X+(size_t)r*D;
    for(j=0;j<D;j++){
      double u0=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+0);
      double u1=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+1);
      double u2=urand(p->seed,key,(uint64_t)i,(uint64_t)j*4+2);
      double xm = toward ? x[j]+u0*(xr[j]-x[j]) : x[j]-u0*(xr[j]-x[j]);   /* Eq.8 */
#if ROUGH_VARIANT==0       /* literal Eq.9: perturbation proportional to the position itself */
      xt[j] = (u2>0.5) ? xm+m*u1*xm : xm-m*u1*xm;
#elif ROUGH_VARIANT==1     /* perturbation proportional to the step just taken (xm - x) */
      { double st=xm-x[j]; xt[j] = (u2>0.5) ? xm+m*u1*st : xm-m*u1*st; }
#else                      /* perturbation proportional to a shrinking radius (my own variant) */
      { double st=0.05*(p->hi-p->lo)*(1.0-(double)k/p->kmax); xt[j] = (u2>0.5) ? xm+m*u1*st : xm-m*u1*st; }
#endif
    }
  }
  for(j=0;j<D;j++) xt[j]=clampd(xt[j],p->lo,p->hi);
  s->Ft[i]=bho_eval(p->fid,xt,s->shift,D);
}

/* ===== STEP 6 (parallel over i): Explorpolis acceptance ======================
 * accept if better than parent; otherwise reject only if some OTHER hunter that
 * is at least as good lies within normalised radius tau (RMS distance / range). */
HD static inline int bho_pair_blocks(const BHOParams *p, const BHOState *s, int i, int j){
  if(j==i || s->F[j]>s->Ft[i]) return 0;
  double range=p->hi-p->lo, thr=p->tau*p->tau*p->D*range*range, d=0.0;
  const double *a=s->Xt+(size_t)i*p->D, *b=s->X+(size_t)j*p->D;
  for(int q=0;q<p->D;q++){ double t=a[q]-b[q]; d+=t*t; if(d>=thr) return 0; }
  return 1;
}
HD static void bho_phase_select(const BHOParams *p, BHOState *s, int i){
  if(s->Ft[i]<s->F[i]){ s->acc[i]=1; return; }
  for(int j=0;j<p->N;j++) if(bho_pair_blocks(p,s,i,j)){ s->acc[i]=0; return; }
  s->acc[i]=1;
}

/* ===== STEP 7 (parallel over i): apply, personal best, score (Eq.5) ========== */
HD static void bho_phase_apply(const BHOParams *p, BHOState *s, int i){
  int D=p->D, j, improved=0; double u=1.0/((double)p->N*p->kmax);
  if(s->acc[i]){
    improved=(s->Ft[i]<s->F[i]);
    for(j=0;j<D;j++) s->X[(size_t)i*D+j]=s->Xt[(size_t)i*D+j];
    s->F[i]=s->Ft[i];
  }
  s->sco[i]=improved?u:-u;
  if(s->F[i]<s->PF[i]){ s->PF[i]=s->F[i]; for(j=0;j<D;j++) s->PX[(size_t)i*D+j]=s->X[(size_t)i*D+j]; }
}

/* ===== STEP 8 (serial): self-feedback r_h (Eq.4), global best, curve ======== */
HD static void bho_phase_finish(const BHOParams *p, BHOState *s, int k, double *curve){
  int N=p->N, i; double rh=s->sc[0];
  for(i=0;i<N;i++){ if(s->mode[i]==0) rh-=s->sco[i]; else rh+=s->sco[i]; }
  s->sc[0]=clampd(rh,0.05,0.95);
  int b=0; for(i=1;i<N;i++) if(s->PF[i]<s->PF[b]) b=i;
  s->ig[0]=b; curve[k]=s->PF[b];
}
#endif

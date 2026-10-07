/* Untrained stateless IEEE-double inference. No native-model/backend coupling. */
#include "kernels.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Finite-input validation requires ordinary IEEE math, without fast-math"
#endif
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
               "This ABI requires IEEE binary64 double");

static int overlap(const double *input,size_t n,const double *output,size_t m) {
    if(!input || !output || n>SIZE_MAX/sizeof(double) || m>SIZE_MAX/sizeof(double))
        return HK_BAD_ARGUMENT;
    uintptr_t a=(uintptr_t)input,b=(uintptr_t)output;
    size_t an=n*sizeof(double),bn=m*sizeof(double);
    if(an>UINTPTR_MAX-a || bn>UINTPTR_MAX-b) return HK_BAD_ARGUMENT;
    return a<b+bn && b<a+an ? HK_OVERLAP : HK_OK;
}

static int finite_array(const double *a,size_t n) {
    for(size_t i=0;i<n;i++) if(!isfinite(a[i])) return HK_NONFINITE;
    return HK_OK;
}

static int common(const double *raw,size_t rn,const double *mean,size_t mn,
                  const double *scale,size_t sn,const double *parameters,size_t pn,
                  size_t expected_p,double *out,size_t on,size_t expected_o) {
    if(!raw || !mean || !scale || !parameters || !out || rn!=HK_INPUTS
       || mn!=HK_INPUTS || sn!=HK_INPUTS || pn!=expected_p || on!=expected_o)
        return HK_BAD_ARGUMENT;
    int status;
    if((status=overlap(raw,rn,out,on)) || (status=overlap(mean,mn,out,on))
       || (status=overlap(scale,sn,out,on)) || (status=overlap(parameters,pn,out,on)))
        return status;
    if((status=finite_array(raw,rn)) || (status=finite_array(mean,mn))
       || (status=finite_array(scale,sn)) || (status=finite_array(parameters,pn)))
        return status;
    for(size_t i=0;i<HK_INPUTS;i++) {
        if(scale[i]<=0) return HK_BAD_SCALE;
        if(raw[i]<0) return HK_NEGATIVE_RAW;
    }
    return HK_OK;
}

static int normalize(const double *raw,const double *mean,const double *scale,double *x) {
    for(size_t i=0;i<HK_INPUTS;i++) {
        double delta=raw[i]-mean[i];
        if(!isfinite(delta)) return HK_ARITHMETIC;
        x[i]=delta/scale[i];
        if(!isfinite(x[i])) return HK_ARITHMETIC;
    }
    return HK_OK;
}

static int add(double a,double b,double *result) {
    double value=a+b;
    if(!isfinite(value)) return HK_ARITHMETIC;
    *result=value;return HK_OK;
}

static int multiply(double a,double b,double *result) {
    double value=a*b;
    if(!isfinite(value)) return HK_ARITHMETIC;
    *result=value;return HK_OK;
}

/* Fixed serial row-major accumulation, with each product and sum checked.
 * No algebraic reassociation or activation is allowed to conceal overflow. */
static int affine(const double *weights,const double *input,size_t width,double bias,double *result) {
    double total=0,term;int status;
    for(size_t j=0;j<width;j++) {
        if((status=multiply(weights[j],input[j],&term)) || (status=add(total,term,&total)))
            return status;
    }
    return add(total,bias,result);
}

static int sigmoid(double a,double *result) {
    /* a is already finite. This evaluates exp only at a nonpositive input. */
    double e=exp(a>=0 ? -a : a);
    if(!isfinite(e)) return HK_ARITHMETIC;
    double denominator=1+e;
    if(!isfinite(denominator) || denominator<=0) return HK_ARITHMETIC;
    double value=a>=0 ? 1/denominator : e/denominator;
    if(!isfinite(value) || value<0 || value>1) return HK_ARITHMETIC;
    *result=value;return HK_OK;
}

static int activate_tanh(double a,double *result) {
    if(!isfinite(a)) return HK_ARITHMETIC;
    double value=tanh(a);
    if(!isfinite(value) || value< -1 || value>1) return HK_ARITHMETIC;
    *result=value;return HK_OK;
}

static int head(const double *weights,const double *hidden,size_t width,double bias,double *out) {
    const double ln2=0x1.62e42fefa39efp-1;
    double scalar,bounded,exponent,factor;int status;
    if((status=affine(weights,hidden,width,bias,&scalar))
       || (status=activate_tanh(scalar,&bounded))
       || (status=multiply(ln2,bounded,&exponent))) return status;
    factor=exp(exponent);
    if(!isfinite(factor) || factor<0.5 || factor>2) return HK_ARITHMETIC;
    out[0]=scalar;out[1]=factor;return HK_OK;
}

int hk_mlp16(const double *raw,size_t raw_n,const double *mean,size_t mean_n,
             const double *scale,size_t scale_n,const double *parameters,size_t parameters_n,
             double *out,size_t out_n) {
    int status=common(raw,raw_n,mean,mean_n,scale,scale_n,parameters,parameters_n,
                      HK_MLP_PARAMETERS,out,out_n,HK_MLP_OUTPUTS);
    if(status) return status;
    double x[HK_INPUTS],candidate[HK_MLP_OUTPUTS],pre;
    if((status=normalize(raw,mean,scale,x))) return status;
    for(size_t i=0;i<HK_MLP_HIDDEN;i++) {
        if((status=affine(parameters+3*i,x,HK_INPUTS,parameters[48+i],&pre))
           || (status=activate_tanh(pre,&candidate[2+i]))) return status;
    }
    if((status=head(parameters+64,candidate+2,HK_MLP_HIDDEN,parameters[80],candidate))) return status;
    memcpy(out,candidate,sizeof(candidate));return HK_OK;
}

int hk_gru32(const double *raw,size_t raw_n,const double *mean,size_t mean_n,
             const double *scale,size_t scale_n,const double *parameters,size_t parameters_n,
             const double *h,size_t h_n,double *out,size_t out_n) {
    if(!h || h_n!=HK_GRU_HIDDEN) return HK_BAD_ARGUMENT;
    int status=common(raw,raw_n,mean,mean_n,scale,scale_n,parameters,parameters_n,
                      HK_GRU_PARAMETERS,out,out_n,HK_GRU_OUTPUTS);
    if(status) return status;
    if((status=overlap(h,h_n,out,out_n)) || (status=finite_array(h,h_n))) return status;
    double x[HK_INPUTS],r[HK_GRU_HIDDEN],z[HK_GRU_HIDDEN],candidate[HK_GRU_OUTPUTS];
    if((status=normalize(raw,mean,scale,x))) return status;
    for(size_t gate=0;gate<2;gate++) {
        double *gate_values=gate==0 ? r : z;
        for(size_t i=0;i<HK_GRU_HIDDEN;i++) {
            size_t row=gate*HK_GRU_HIDDEN+i;
            double input,recurrent,pre;
            if((status=affine(parameters+3*row,x,HK_INPUTS,parameters[3360+row],&input))
               || (status=affine(parameters+288+32*row,h,HK_GRU_HIDDEN,0,&recurrent))
               || (status=add(input,recurrent,&pre))
               || (status=sigmoid(pre,&gate_values[i]))) return status;
        }
    }
    for(size_t i=0;i<HK_GRU_HIDDEN;i++) {
        size_t row=2*HK_GRU_HIDDEN+i;
        double input,recurrent,reset_recurrent,pre,n,left,right;
        if((status=affine(parameters+3*row,x,HK_INPUTS,parameters[3360+row],&input))
           || (status=affine(parameters+288+32*row,h,HK_GRU_HIDDEN,0,&recurrent))
           || (status=multiply(r[i],recurrent,&reset_recurrent))
           || (status=add(input,reset_recurrent,&pre))
           || (status=activate_tanh(pre,&n))) return status;
        double one_minus_z=1-z[i];
        if(!isfinite(one_minus_z)) return HK_ARITHMETIC;
        if((status=multiply(one_minus_z,n,&left)) || (status=multiply(z[i],h[i],&right))
           || (status=add(left,right,&candidate[2+i]))) return status;
    }
    if((status=head(parameters+3456,candidate+2,HK_GRU_HIDDEN,parameters[3488],candidate))) return status;
    memcpy(out,candidate,sizeof(candidate));return HK_OK;
}

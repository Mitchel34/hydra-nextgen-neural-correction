/* Immutable-model neural-demand wrapper. CFE alone owns physical state. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "neural_bridge.h"
#include "kernels.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Neural bridge validation requires ordinary finite IEEE math"
#endif
_Static_assert(sizeof(double)==8 && DBL_MANT_DIG==53 && DBL_MAX_EXP==1024,
               "Neural artifact v1 requires IEEE binary64");

#ifdef HYDRA_NN_SOURCE_FIXTURE
extern Bmi *nn_fixture_backend_register(Bmi *);
#endif

typedef struct Context {
    Bmi *self;
    Bmi original;
    void *library;
    int native_initialized, busy, initialize_attempted;
    double staged[2], export_values[70];
    double means[3], scales[3], parameters[HK_GRU_PARAMETERS];
    size_t parameter_count;
    HnFrame frame;
    HnStatus status;
    struct Context *next;
} Context;
static Context *instances;
static uint64_t sequence;
static const char *PET="water_potential_evaporation_flux";
static const char *RAIN="atmosphere_water__liquid_equivalent_precipitation_rate";
static const char *aliases[]={HN_RAW_DEMAND,HN_RAW_LIQUID,HN_SOIL_PRE,HN_INTERVAL_START,
                              "HYDRA_NN_FACTOR","HYDRA_NN_HEAD"};
static const char *units[]={"m s-1","mm h-1","m","s","1","1"};
static const char *h_prefix[]={"HYDRA_NN_H_PRE_","HYDRA_NN_H_POST_"};

static Context *context(Bmi *self) {
    for(Context *c=instances;c;c=c->next) if(c->self==self) return c;
    return NULL;
}
static int alias(const char *name) {
    if(!name) return -1;
    for(int i=0;i<6;i++) if(!strcmp(name,aliases[i])) return i;
    for(int side=0;side<2;side++) {
        size_t n=strlen(h_prefix[side]);
        if(strlen(name)==n+2 && !strncmp(name,h_prefix[side],n)
           && name[n]>='0' && name[n]<='9' && name[n+1]>='0' && name[n+1]<='9') {
            int index=10*(name[n]-'0')+name[n+1]-'0';
            if(index<32) return 6+32*side+index;
        }
    }
    return -1;
}
static int alias_count(Context *c) { return c->status.architecture==HN_GRU ? 70 : 6; }
static int alias_available(Context *c,int k) {
    return c->status.initialized && k>=0 && k<alias_count(c);
}
static void alias_name(int k,char *name) {
    if(k<6) strcpy(name,aliases[k]);
    else snprintf(name,BMI_MAX_VAR_NAME,"%s%02d",h_prefix[(k-6)/32],(k-6)%32);
}
static int forcing(const char *name) {
    return !name ? -1 : !strcmp(name,PET) ? 0 : !strcmp(name,RAIN) ? 1 : -1;
}
static int poison(Context *c) {
    if(c) { c->status.poisoned=1; c->status.ready_mask=0; c->busy=0; }
    return BMI_FAILURE;
}
static int clock_is(Context *c,double expected) {
    double now=NAN;
    return c->original.get_current_time(c->self,&now)==BMI_SUCCESS
        && isfinite(now) && now==expected;
}
/* OS fork copies both native and wrapper state. Only clean boundaries qualify. */
static int owner(Context *c) {
    int64_t pid=(int64_t)getpid();
    if(c->status.owner_pid==pid) return !c->status.poisoned;
    if(!c->status.initialized || c->status.poisoned || c->busy || c->status.ready_mask
       || !clock_is(c,c->status.expected_time_seconds)) { poison(c); return 0; }
    c->status.parent_pid=c->status.owner_pid;
    c->status.owner_pid=pid; c->status.fork_generation++;
    return 1;
}
static int usable(Context *c) {
    return c && c->status.initialized && !c->status.poisoned && owner(c);
}
static int scalar(Context *c,const char *name,const char *expected_units) {
    char type[BMI_MAX_TYPE_NAME]={0},actual[BMI_MAX_UNITS_NAME]={0};
    int size=-1,nbytes=-1;
    return c->original.get_var_type(c->self,name,type)==BMI_SUCCESS
        && !strcmp(type,"double")
        && c->original.get_var_units(c->self,name,actual)==BMI_SUCCESS
        && !strcmp(actual,expected_units)
        && c->original.get_var_itemsize(c->self,name,&size)==BMI_SUCCESS && size==8
        && c->original.get_var_nbytes(c->self,name,&nbytes)==BMI_SUCCESS && nbytes==8;
}
static int copy_scalar(Context *c,const char *name,double *value) {
    *value=NAN;
    return c->original.get_value(c->self,name,value)==BMI_SUCCESS
        && isfinite(*value) && *value>=0;
}
static int same_double(double a,double b) { return memcmp(&a,&b,sizeof(double))==0; }
static void exports(Context *c) {
    c->export_values[0]=c->frame.raw_demand_m_s;
    c->export_values[1]=c->frame.raw_liquid_mm_h;
    c->export_values[2]=c->frame.soil_before_m;
    c->export_values[3]=c->frame.interval_start_seconds;
    c->export_values[4]=c->frame.factor;
    c->export_values[5]=c->frame.head;
    for(int i=0;i<32;i++) {
        c->export_values[6+i]=c->frame.h_pre[i];
        c->export_values[38+i]=c->frame.h_post[i];
    }
}
static int stable_stat(const struct stat *a,const struct stat *b) {
    if(a->st_dev!=b->st_dev || a->st_ino!=b->st_ino || a->st_size!=b->st_size
       || a->st_mtime!=b->st_mtime || a->st_ctime!=b->st_ctime) return 0;
#ifdef __APPLE__
    return a->st_mtimespec.tv_nsec==b->st_mtimespec.tv_nsec
        && a->st_ctimespec.tv_nsec==b->st_ctimespec.tv_nsec;
#else
    return a->st_mtim.tv_nsec==b->st_mtim.tv_nsec && a->st_ctim.tv_nsec==b->st_ctim.tv_nsec;
#endif
}
static int read_regular(const char *path,unsigned char *bytes,size_t capacity,size_t *length) {
    if(!path || path[0]!='/') return 0;
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK);if(fd<0) return 0;
    struct stat before,after;int ok=0;
    if(fstat(fd,&before) || !S_ISREG(before.st_mode) || before.st_size<=0
       || (uint64_t)before.st_size>capacity) goto done;
    size_t used=0,wanted=(size_t)before.st_size;
    while(used<wanted) {
        ssize_t n=read(fd,bytes+used,wanted-used);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) goto done;
        used+=(size_t)n;
    }
    unsigned char trailing;ssize_t n;
    do { n=read(fd,&trailing,1); } while(n<0 && errno==EINTR);
    if(n!=0 || fstat(fd,&after) || !stable_stat(&before,&after)) goto done;
    *length=used;ok=1;
done:
    close(fd);return ok;
}
static int hex64(const char *value) {
    return strlen(value)==64 && strspn(value,"0123456789abcdef")==64;
}
static int parse_config(Context *c,const char *file,char *path,char *model_path) {
    unsigned char bytes[32769];size_t size=0;
    if(!read_regular(file,bytes,sizeof(bytes)-1,&size) || memchr(bytes,0,size)) return 0;
    bytes[size]=0;char *line=(char *)bytes;unsigned seen=0;
    while(line) {
        char *next=strchr(line,'\n');if(next) *next++=0;
        size_t n=strlen(line);if(n && line[n-1]=='\r') line[n-1]=0;
        if(!line[0] || line[0]=='#') { line=next;continue; }
        char *eq=strchr(line,'=');if(!eq) return 0;
        *eq++=0;unsigned bit=0;
        if(!strcmp(line,"cfe_config_file") || !strcmp(line,"model_artifact_file")) {
            int model=!strcmp(line,"model_artifact_file");bit=model ? 16 : 1;
            if(eq[0]!='/' || strlen(eq)>=4096) return 0;
            strcpy(model ? model_path : path,eq);
        } else if(!strcmp(line,"catchment_id")) {
            bit=2;size_t len=strlen(eq);
            if(!len || len>=HN_CATCHMENT_CAP || strspn(eq,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=len) return 0;
            strcpy(c->frame.catchment_id,eq);
        } else if(!strcmp(line,"config_sha256") || !strcmp(line,"model_file_sha256")) {
            int model=!strcmp(line,"model_file_sha256");bit=model ? 32 : 4;
            if(!hex64(eq)) return 0;
            strcpy(model ? c->frame.model_file_sha256 : c->frame.config_sha256,eq);
        } else if(!strcmp(line,"origin_epoch")) {
            bit=8;errno=0;char *end=NULL;long long epoch=strtoll(eq,&end,10);
            if(errno || !*eq || *end || epoch<0 || epoch%3600) return 0;
            c->frame.origin_epoch=(int64_t)epoch;
        } else return 0;
        if(seen&bit) return 0;
        seen|=bit;line=next;
    }
    return seen==63;
}
static uint32_t little_u32(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static double little_f64(const unsigned char *p) {
    uint64_t bits=0;for(unsigned i=0;i<8;i++) bits|=(uint64_t)p[i]<<(8*i);
    double value;memcpy(&value,&bits,8);return value;
}
static int load_model(Context *c,const char *path) {
    unsigned char bytes[HN_ARTIFACT_HEADER_BYTES+8*(6+HK_GRU_PARAMETERS)];size_t size=0;
    if(!read_regular(path,bytes,sizeof(bytes),&size) || size<HN_ARTIFACT_HEADER_BYTES
       || memcmp(bytes,"HYDRANN1",8)) return 0;
    uint32_t fields[9];for(int i=0;i<9;i++) fields[i]=little_u32(bytes+8+4*i);
    uint32_t architecture=fields[2],hidden=architecture==HN_MLP ? 16 : 32;
    uint32_t count=architecture==HN_MLP ? HK_MLP_PARAMETERS : HK_GRU_PARAMETERS;
    if(fields[0]!=1 || fields[1]!=HN_ARTIFACT_HEADER_BYTES || (architecture!=HN_MLP && architecture!=HN_GRU)
       || fields[3]!=3 || fields[4]!=hidden || fields[5]!=count || fields[6]!=1
       || fields[7]!=(architecture==HN_MLP ? 0u : 1u) || fields[8]!=count+6
       || size!=HN_ARTIFACT_HEADER_BYTES+8*(size_t)(count+6)) return 0;
    char tags[4][65];for(int i=0;i<4;i++) { memcpy(tags[i],bytes+44+64*i,64);tags[i][64]=0;if(!hex64(tags[i])) return 0; }
    if(strcmp(tags[0],HN_FEATURE_ID) || strcmp(tags[1],architecture==HN_MLP ? HN_MLP_LAYOUT_ID : HN_GRU_LAYOUT_ID)) return 0;
    for(uint32_t i=0;i<count+6;i++) {
        double value=little_f64(bytes+HN_ARTIFACT_HEADER_BYTES+8*i);
        if(!isfinite(value) || (i>=3 && i<6 && value<=0)) return 0;
        if(i<3) c->means[i]=value;
        else if(i<6) c->scales[i-3]=value;
        else c->parameters[i-6]=value;
    }
    c->parameter_count=count;c->frame.architecture=c->status.architecture=architecture;
    c->frame.hidden_count=c->status.hidden_count=architecture==HN_GRU ? 32 : 0;
    strcpy(c->frame.feature_id,tags[0]);strcpy(c->frame.layout_id,tags[1]);
    strcpy(c->frame.model_id,tags[2]);strcpy(c->frame.scaler_id,tags[3]);
    return 1;
}
static int bridge_initialize(Bmi *self,const char *file) {
    Context *c=context(self);
    if(!c || !file || c->initialize_attempted || c->status.poisoned) return poison(c);
    c->initialize_attempted=1;
    char path[4096]={0},model_path[4096]={0};
    if(!parse_config(c,file,path,model_path) || !load_model(c,model_path)) return poison(c);
    /* A failed opaque Initialize does not establish a safe native Finalize. */
    if(c->original.initialize(self,path)!=BMI_SUCCESS) return poison(c);
    c->native_initialized=1;
    double dt=NAN,soil=NAN; char time_units[BMI_MAX_UNITS_NAME]={0};
    if(c->original.get_time_step(self,&dt)!=BMI_SUCCESS || dt!=3600
       || c->original.get_time_units(self,time_units)!=BMI_SUCCESS || strcmp(time_units,"s")
       || !clock_is(c,0) || !scalar(c,PET,units[0]) || !scalar(c,RAIN,units[1])
       || !scalar(c,"SOIL_STORAGE",units[2]) || !copy_scalar(c,"SOIL_STORAGE",&soil)) return poison(c);
    memset(c->status.committed_h,0,sizeof(c->status.committed_h));
    c->status.initialized=1; c->status.expected_time_seconds=0;
    return BMI_SUCCESS;
}
static int bridge_set(Bmi *self,const char *name,void *src) {
    Context *c=context(self);
    if(!usable(c) || !name || !src || alias(name)>=0) return poison(c);
    int k=forcing(name);
    if(k>=0) {
        double value; memcpy(&value,src,sizeof(value));
        if((c->status.ready_mask&(1u<<k)) || !clock_is(c,c->status.expected_time_seconds)
           || !scalar(c,name,units[k]) || !isfinite(value) || value<0) return poison(c);
        if(c->original.set_value(self,name,src)!=BMI_SUCCESS) return poison(c);
        c->staged[k]=value; c->status.ready_mask|=1u<<k; c->status.feature_input_sets++;
        return BMI_SUCCESS;
    }
    int rc=c->original.set_value(self,name,src);
    return rc==BMI_SUCCESS ? rc : poison(c);
}
static int bridge_set_indices(Bmi *self,const char *name,int *indices,int count,void *src) {
    Context *c=context(self);
    if(!usable(c) || !name || !src || alias(name)>=0) return poison(c);
    if(forcing(name)>=0) {
        if(!indices || count!=1 || indices[0]!=0) return poison(c);
        return bridge_set(self,name,src);
    }
    int rc=c->original.set_value_at_indices(self,name,indices,count,src);
    return rc==BMI_SUCCESS ? rc : poison(c);
}
static int bridge_update(Bmi *self) {
    Context *c=context(self);
    if(!usable(c) || c->busy || c->status.ready_mask!=3
       || !clock_is(c,c->status.expected_time_seconds)) return poison(c);
    c->status.capture_attempts++;
    double pet=NAN,rain=NAN,soil=NAN;
    if(!scalar(c,PET,units[0]) || !scalar(c,RAIN,units[1]) || !scalar(c,"SOIL_STORAGE",units[2])
       || !copy_scalar(c,PET,&pet) || !copy_scalar(c,RAIN,&rain)
       || !same_double(pet,c->staged[0]) || !same_double(rain,c->staged[1])
       || !copy_scalar(c,"SOIL_STORAGE",&soil)) return poison(c);
    HnFrame candidate=c->frame;
    double now=c->status.expected_time_seconds;
    /* Epoch addition must remain integral and representable without overflow. */
    if(now<0 || now>9007199254740991.0 || now!=floor(now)
       || c->frame.origin_epoch>INT64_MAX-(int64_t)now) return poison(c);
    candidate.valid=1; candidate.frame_index=c->status.successful_updates+1;
    candidate.interval_start_seconds=now;
    candidate.interval_start_epoch=c->frame.origin_epoch+(int64_t)now;
    candidate.raw_demand_m_s=pet; candidate.raw_liquid_mm_h=rain; candidate.soil_before_m=soil;
    c->busy=1;
    double raw[3]={pet,rain,soil},neural[HK_GRU_OUTPUTS];
    for(size_t i=0;i<HK_GRU_OUTPUTS;i++) neural[i]=NAN;
    c->status.inference_calls++;
    int kernel;
    if(c->status.architecture==HN_MLP) {
        kernel=hk_mlp16(raw,3,c->means,3,c->scales,3,c->parameters,c->parameter_count,
                        neural,HK_MLP_OUTPUTS);
    } else {
        kernel=hk_gru32(raw,3,c->means,3,c->scales,3,c->parameters,c->parameter_count,
                        c->status.committed_h,32,neural,HK_GRU_OUTPUTS);
    }
    c->status.last_kernel_status=kernel;
    if(kernel!=HK_OK || !isfinite(neural[0]) || !isfinite(neural[1])
       || neural[1]<0.5 || neural[1]>2) return poison(c);
    candidate.head=neural[0];candidate.factor=neural[1];
    candidate.applied_demand_m_s=pet*candidate.factor;
    if(!isfinite(candidate.applied_demand_m_s) || candidate.applied_demand_m_s<0) return poison(c);
    if(c->status.architecture==HN_GRU) {
        for(int i=0;i<32;i++) {
            if(!isfinite(neural[2+i])) return poison(c);
            candidate.h_pre[i]=c->status.committed_h[i];candidate.h_post[i]=neural[2+i];
        }
    }
    if(candidate.factor!=1.0) {
        /* Original PET getters still address native CFE storage. This original
         * setter is one correction write following the unchanged raw staging
         * write, not a second multiplication or a physical Update. */
        c->status.demand_override_calls++;
        if(c->original.set_value(self,PET,&candidate.applied_demand_m_s)!=BMI_SUCCESS) return poison(c);
    }
    double applied_read=NAN;
    if(!copy_scalar(c,PET,&applied_read) || !same_double(applied_read,candidate.applied_demand_m_s)) return poison(c);
    c->status.native_update_calls++;
    int rc=c->original.update(self);
    if(rc!=BMI_SUCCESS || !clock_is(c,now+3600) || !copy_scalar(c,"SOIL_STORAGE",&soil)) return poison(c);
    if(c->status.architecture==HN_GRU) memcpy(c->status.committed_h,candidate.h_post,sizeof(c->status.committed_h));
    c->frame=candidate; c->status.successful_updates++;
    c->status.frame_valid=1; c->status.expected_time_seconds=now+3600;
    c->status.ready_mask=0; c->busy=0; exports(c);
    return BMI_SUCCESS;
}
static int bridge_until(Bmi *self,double then) {
    Context *c=context(self);
    if(!usable(c) || !isfinite(then) || then!=c->status.expected_time_seconds+3600) return poison(c);
    return bridge_update(self);
}
static int diagnostic(Context *c) { return usable(c) && c->frame.valid; }
static int bridge_get(Bmi *self,const char *name,void *dest) {
    Context *c=context(self); if(!c || !name || !dest) return BMI_FAILURE;
    int k=alias(name);
    if(k<0) return c->original.get_value(self,name,dest);
    if(!alias_available(c,k) || !diagnostic(c)) return BMI_FAILURE;
    exports(c); memcpy(dest,&c->export_values[k],sizeof(double)); c->status.frame_reads++;
    return BMI_SUCCESS;
}
static int bridge_ptr(Bmi *self,const char *name,void **dest) {
    Context *c=context(self); if(!c || !name || !dest) return BMI_FAILURE;
    int k=alias(name);
    if(k<0) return c->original.get_value_ptr(self,name,dest);
    *dest=NULL; if(!alias_available(c,k) || !diagnostic(c)) return BMI_FAILURE;
    exports(c); *dest=&c->export_values[k]; c->status.pointer_exports++;
    return BMI_SUCCESS;
}
static int bridge_indices(Bmi *self,const char *name,void *dest,int *indices,int count) {
    Context *c=context(self); if(!c || !name || !dest) return BMI_FAILURE;
    if(alias(name)<0) return c->original.get_value_at_indices(self,name,dest,indices,count);
    if(!indices || count!=1 || indices[0]!=0) return BMI_FAILURE;
    return bridge_get(self,name,dest);
}
static int output_count(Bmi *self,int *count) {
    Context *c=context(self); int n=-1;
    if(!c || !c->status.initialized || !count || c->original.get_output_item_count(self,&n)!=BMI_SUCCESS || n<0 || n>1000) return BMI_FAILURE;
    *count=n+alias_count(c); return BMI_SUCCESS;
}
static int output_names(Bmi *self,char **names) {
    Context *c=context(self); int n=-1;
    if(!c || !c->status.initialized || !names || c->original.get_output_item_count(self,&n)!=BMI_SUCCESS || n<0 || n>1000
       || c->original.get_output_var_names(self,names)!=BMI_SUCCESS) return BMI_FAILURE;
    for(int i=0;i<alias_count(c);i++) alias_name(i,names[n+i]);
    return BMI_SUCCESS;
}
#define META_INT(FIELD,NAME,VALUE) \
static int NAME(Bmi *self,const char *name,int *out) { \
    Context *c=context(self); if(!c || !name || !out) return BMI_FAILURE; \
    int k=alias(name);if(k<0) return c->original.FIELD(self,name,out); \
    if(!alias_available(c,k)) return BMI_FAILURE; \
    *out=(VALUE); return BMI_SUCCESS; }
META_INT(get_var_grid,meta_grid,0)
META_INT(get_var_itemsize,meta_itemsize,8)
META_INT(get_var_nbytes,meta_nbytes,8)
static int meta_type(Bmi *self,const char *name,char *out) {
    Context *c=context(self); if(!c || !name || !out) return BMI_FAILURE;
    int k=alias(name);if(k<0) return c->original.get_var_type(self,name,out);
    if(!alias_available(c,k)) return BMI_FAILURE;
    strcpy(out,"double"); return BMI_SUCCESS;
}
static int meta_units(Bmi *self,const char *name,char *out) {
    Context *c=context(self); if(!c || !name || !out) return BMI_FAILURE;
    int k=alias(name); if(k<0) return c->original.get_var_units(self,name,out);
    if(!alias_available(c,k)) return BMI_FAILURE;
    strcpy(out,k<6 ? units[k] : "1"); return BMI_SUCCESS;
}
static int meta_location(Bmi *self,const char *name,char *out) {
    Context *c=context(self); if(!c || !name || !out) return BMI_FAILURE;
    int k=alias(name);if(k<0) return c->original.get_var_location(self,name,out);
    if(!alias_available(c,k)) return BMI_FAILURE;
    strcpy(out,"node"); return BMI_SUCCESS;
}
static int bridge_finalize(Bmi *self) {
    Context **link=&instances; while(*link && (*link)->self!=self) link=&(*link)->next;
    if(!*link) return BMI_FAILURE;
    Context *c=*link; int rc=BMI_SUCCESS;
    if(c->native_initialized) rc=c->original.finalize(self);
    /* Opaque partially initialized allocations are deliberately not finalized. */
    *link=c->next; if(c->library) dlclose(c->library); free(c); return rc;
}
int hydra_neural_snapshot(Bmi *self,HnFrame *frame,HnStatus *status) {
    Context *c=context(self); if(!c || !frame || !status) return BMI_FAILURE;
    int ok=diagnostic(c); c->status.frame_reads++;
    *frame=c->frame; *status=c->status;
    return ok ? BMI_SUCCESS : BMI_FAILURE;
}
size_t hydra_neural_frame_size(void) { return sizeof(HnFrame); }
size_t hydra_neural_status_size(void) { return sizeof(HnStatus); }
Bmi *register_bmi_hydra_neural(Bmi *self) {
    if(!self || context(self)) return NULL;
    Context *c=calloc(1,sizeof(*c)); if(!c) return NULL;
    Bmi *(*registration)(Bmi *)=NULL;
#ifdef HYDRA_NN_SOURCE_FIXTURE
    registration=nn_fixture_backend_register;
#else
    const char *path=getenv("HYDRA_CFE_LIBRARY");
    if(!path || path[0]!='/') { free(c); return NULL; }
    c->library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!c->library) { free(c); return NULL; }
    *(void **)(&registration)=dlsym(c->library,"register_bmi_cfe");
    if(!registration) { dlclose(c->library); free(c); return NULL; }
#endif
    if(!registration(self)) { if(c->library) dlclose(c->library); free(c); return NULL; }
    c->self=self; c->original=*self;
    c->frame.abi_version=HN_ABI_VERSION;
    c->frame.raw_demand_m_s=c->frame.raw_liquid_mm_h=c->frame.soil_before_m=NAN;
    c->frame.interval_start_seconds=NAN;
    c->frame.head=c->frame.factor=c->frame.applied_demand_m_s=NAN;
    c->status.abi_version=HN_ABI_VERSION; c->status.instance_sequence=++sequence;
    c->status.created_pid=c->status.owner_pid=(int64_t)getpid();
    c->status.expected_time_seconds=NAN;
    c->status.last_kernel_status=-1;
    c->next=instances; instances=c;
    self->initialize=bridge_initialize; self->update=bridge_update; self->update_until=bridge_until;
    self->finalize=bridge_finalize; self->set_value=bridge_set; self->set_value_at_indices=bridge_set_indices;
    self->get_value=bridge_get; self->get_value_ptr=bridge_ptr; self->get_value_at_indices=bridge_indices;
    self->get_output_item_count=output_count; self->get_output_var_names=output_names;
    self->get_var_grid=meta_grid; self->get_var_type=meta_type; self->get_var_units=meta_units;
    self->get_var_itemsize=meta_itemsize; self->get_var_nbytes=meta_nbytes; self->get_var_location=meta_location;
    return self;
}

/* Verification-only I/O glue. Every arithmetic operation calls existing
 * seeded-Omega kernels from ggml-omega e0813903. No model or GPU is opened. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error little-endian verification target required
#endif
uint32_t omega_bin_32(uint64_t *,uint64_t,const void *,uint64_t,const void *,uint64_t,void *,uint64_t);
uint32_t omega_bin_1(uint64_t *,uint64_t,const void *,uint64_t,const void *,uint64_t,void *,uint64_t);
uint32_t omega_sexp_32(uint64_t *,uint64_t,void *,uint64_t,uint64_t);
uint32_t omega_sexp_1(uint64_t *,uint64_t,void *,uint64_t,uint64_t);
int main(int argc,char **argv) {
    _Static_assert(sizeof(float)==4,"requires float32");
    if(argc!=2 || (strcmp(argv[1],"plus1") && strcmp(argv[1],"negexp")))return 2;
    unsigned char raw[65536];
    size_t bytes=fread(raw,1,sizeof raw,stdin);
    if(ferror(stdin)||!feof(stdin)||!bytes||bytes%4)return 2;
    float data[16384],out[16384],ones[32];memcpy(data,raw,bytes);
    for(size_t j=0;j<32;j++)ones[j]=1.0f;
    const size_t n=bytes/4;size_t i=0;uint64_t r=0;uint32_t status;
    while(i<n){const size_t lanes=n-i>=32?32:1;
        if(!strcmp(argv[1],"plus1")) {
            status=lanes==32?omega_bin_32(&r,0,data+i,32,ones,32,out+i,32):omega_bin_1(&r,0,data+i,1,ones,1,out+i,1);
        } else {
            status=lanes==32?omega_sexp_32(&r,0,data+i,32,0):omega_sexp_1(&r,0,data+i,1,0);
            for(size_t j=0;j<lanes;j++){uint32_t u;memcpy(&u,data+i+j,4);u^=UINT32_C(0x80000000);memcpy(out+i+j,&u,4);}
        }
        if(status)return 3;
        i+=lanes;
    }
    return fwrite(out,1,bytes,stdout)==bytes?0:4;
}

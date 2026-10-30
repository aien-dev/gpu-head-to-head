// Independent verification oracle only. Uses the original converter's pinned
// ATen CPU exp, without Python, model code, CUDA computation, or a production dependency.
// The pinned library's dependencies include libcudart; no library-absence claim is made.
#include <ATen/ATen.h>
#include <cstdio>
#include <cstring>
int main(int argc,char **argv) {
    if(argc!=2||std::strcmp(argv[1],"negexp"))return 2;
    static_assert(sizeof(float)==4);
    unsigned char raw[65536];const size_t bytes=std::fread(raw,1,sizeof raw,stdin);
    if(std::ferror(stdin)||!std::feof(stdin)||!bytes||bytes%4)return 2;
    float values[16384];std::memcpy(values,raw,bytes);const size_t n=bytes/4;
    try {
        auto input=at::from_blob(values,{static_cast<int64_t>(n)},at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
        auto result=at::neg(at::exp(input)).contiguous();
        return std::fwrite(result.const_data_ptr<float>(),sizeof(float),n,stdout)==n?0:4;
    }catch(const std::exception &){return 3;}
}

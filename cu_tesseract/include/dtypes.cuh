#ifndef DTYPES_HPP
#define DTYPES_HPP

#include <cstddef>
#include <cuda_fp16.h>

typedef float fp32;
typedef double fp64;
typedef half fp16;
typedef __nv_bfloat16 bf16;

using std::size_t;

#endif
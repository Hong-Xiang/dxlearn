#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"

struct TOF_LOR
{
    // start point
    float *x1;
    float *y1;
    float *z1;
    // end point
    float *x2;
    float *y2;
    float *z2;
    // tof center
    float *xc;
    float *yc;
    float *zc;
    // tof distance(t2 > t1, so t_dis = light_speed*(t2-t1)/2)
    // which indicates the center position
    float *t_dis;
}

/// calculate the center of tof lor, and update the two ends according to the 
/// 
__device__ void scissor(float* x1, float* y1, float* z1,
                        float* x2, float* y2, float* z2,
                        float* xc, float* yc, float* zc,
                        const float* t_dis, const float limit)   
{
    //
    float dx = *x2 - *x1, dy = *y2 - *y1, dz = *z2 - *z1;
    float len =  std::sqrt(dx*dx + dy*dy + dz*dz);
    float 



}
__global__ void process(const TOF_LOR *tlor, int num_event, float limit)
{
    float *x1 = tlor->x1;
    float *y1 = tlor->y1;
    float *z1 = tlor->z1;
    float *x2 = tlor->x2;
    float *y2 = tlor->y2;
    float *z2 = tlor->z2;
    float *xc = tlor->xc;
    float *yc = tlor->yc;
    float *zc = tlor->zc;
    float *t_dis = tlor->t_dis
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_events;
         tid += blockDim.x * gridDim.x)
    {
        scissor(x1 + tid, y1 + tid, z1 + tid,
                x2 + tid, y2 + tid, z2 + tid,
                xc + tid, yc + tid, zc + tid, 
                t_dis + tid, limit);
    }
}


///
void cut_lors(const TOF_LOR *tlor, int num_lors, float limit)
{
    //

    //
}
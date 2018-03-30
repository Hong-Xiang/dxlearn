#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"


struct TOF_LOR{
    // start point
    float* x1;
    float* y1;
    float* z1;
    // end point
    float* x2;
    float* y2;
    float* z2;
    // tof center
    float* xc;
    float* yc;
    float* zc;
    // tof distance(t2 > t1, so t_dis = light_speed*(t2-t1)/2)
    // which indicates the center position 
    float* t_dis;
}

/// cut the lors according to the tof information.
/// this operation reduce the lor length due to the tof kernel
///
void cut_lors( const TOF_LOR* tlor, )
{
    // 

    //

}
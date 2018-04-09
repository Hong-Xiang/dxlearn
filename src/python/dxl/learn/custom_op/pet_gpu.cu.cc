#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cmath>

// const float LN2 = M_LN2;
///
/// calculate the cross point 
__device__ bool CalculateCrossPoint(float pos1_x, float pos1_y, float pos1_z,
                                    float pos2_x, float pos2_y, float pos2_z,
                                    const float s_center,
                                    float &dcos_x, float &dcos_y,
                                    float &cross_x, float &cross_y)
{
    float d0 = pos2_x - pos1_x, d1 = pos2_y - pos1_y, d2 = pos2_z - pos1_z;
    float ratio = (s_center - pos1_z) / d2;
    if (ratio < 0.0 || ratio > 1.0)
        return false;
    //the x and y value of cross point.
    cross_x = pos1_x + d0 * ratio;
    cross_y = pos1_y + d1 * ratio;
    //the len of lor.
    float dis = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    //direction cosine of lor.
    dcos_x = d0 / dis;
    dcos_y = d1 / dis;
    // printf("calculate points: %f \n",s_center);
    return true;
}

__device__ void CalculateSMV(const float cross_x, const float cross_y,
                             const float mesh_x, const float mesh_y,
                             const float dcos_x, const float dcos_y,
                             const float sigma2, float &value)
{
    float delta_x = cross_x - mesh_x;
    float delta_y = cross_y - mesh_y;
    float r_cos = (delta_x * dcos_x + delta_y * dcos_y);
    float d2 = delta_x * delta_x + delta_y * delta_y - r_cos * r_cos;
    value = (d2 < 9.0 * sigma2) ? std::exp(-0.5 * d2 / sigma2) : 0.0;
}

__device__ void LoopPatch(const unsigned patch_size, const unsigned int offset,
                          const float inter_x, const float inter_y,
                          const float cross_x, const float cross_y,
                          const float sigma2, const float dcos_x, const float dcos_y,
                          const float l_bound, const float b_bound, const int l0, const int l1,
                          float *projection_value, const float *image_data)
{
    // auto image_flat = image.flat<float>();
    // int l0 = image.dim_size(0);
    // int l1 = image.dim_size(1);
    //the start mesh of this current patch
    int index_x = std::round((cross_x - l_bound) / inter_x) - (int)(patch_size / 2);
    int index_y = std::round((cross_y - b_bound) / inter_y) - (int)(patch_size / 2);
    for (int j = 0; j < patch_size; j++)
    {
        int index1 = index_y + j;
        if ( index1 < 0 || index1 >= l1 ) //y axis index is out of slice range
            continue;    
        for (int i = 0; i < patch_size; i++)
        {
            int index0 = index_x + i;
            if (index0 < 0 || index0 >= l0 ) // x axis index is out of slice range
                continue;
            else
            {
                int index = index0 + index1 * l0;
                float value = 0.0;
                // compute the system matrix value.
                CalculateSMV(cross_x, cross_y,
                             inter_x * (float)(0.5 + index0) + l_bound,
                             inter_y * (float)(0.5 + index1) + b_bound,
                             dcos_x, dcos_y, sigma2, value);
                // printf("SMV value: %f \n", value);
                atomicAdd(projection_value,image_data[offset +index]* value);
            }
        }
    }
}

// for backprojection
__device__ void BackLoopPatch(const unsigned patch_size, const unsigned int offset,
                              const float inter_x, const float inter_y,
                              const float cross_x, const float cross_y,
                              const float sigma2, const float dcos_x, const float dcos_y,
                              const float l_bound, const float b_bound, const int l0, const int l1,
                              const float projection_value, float *image_data)
{
    // int index_x = (int)((cross_x - l_bound) / inter_x) - (int)(patch_size / 2);
    // int index_y = (int)((cross_y - b_bound) / inter_y) - (int)(patch_size / 2);
    int index_x = (int)((cross_x - l_bound) / inter_x ) - (int)(patch_size / 2 );
    int index_y = (int)((cross_y - b_bound) / inter_y ) - (int)(patch_size / 2 );
    for (int j = 0; j < patch_size; j++)
    {
        int index1 = index_y + j;
        if ( index1 < 0 || index1 >= l1) //y axis index is out of slice range
            continue;    
        for (int i = 0; i < patch_size; i++)
        {
            int index0 = index_x + i;
            if (index0 < 0 || index0 >= l0) // x axis index is out of slice range
                continue;
            else
            {
                int index = index0 + index1 * l0;
                float value = 0.0;
                // compute the system matrix value.
                CalculateSMV(cross_x, cross_y,
                             inter_x * (float)(0.5 + index0) + l_bound,
                             inter_y * (float)(0.5 + index1) + b_bound,
                            //  inter_x * (index0 ) + l_bound,
                            //  inter_y * (index1 ) + b_bound,
                             dcos_x, dcos_y, sigma2, value);
                if (projection_value > 1e-7)
                    atomicAdd(image_data + offset + index, value / projection_value);
            }
        }
    }
}

__global__ void ComputeSlice(const float *x1, const float *y1, const float *z1,
                             const float *x2, const float *y2, const float *z2,
                             const unsigned int patch_size, const unsigned int offset,
                             const float slice_z,
                             const float l_bound, const float b_bound, const float sigma2,
                             const int gx, const int gy, const float inter_x, const float inter_y,
                             float *projection_value, const int num_events, const float *image)
{
    // int counter = 0;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_events;
         tid += blockDim.x * gridDim.x)
    {
        // unsigned int num_events = events.dim_size(0);
        //std::cout<<"events dims:"<<events.dim_size(0)<<", "<<events.dim_size(1)<<std::endl;
        //debug
        //std::cout<<"event:"<<event.IsAligned()<<std::endl;
        float dcos_x = 0;
        float dcos_y = 0;
        float cross_x = 0;
        float cross_y = 0;
        if (CalculateCrossPoint(x1[tid], y1[tid], z1[tid],
                                x2[tid], y2[tid], z2[tid],
                                slice_z, dcos_x, dcos_y, cross_x, cross_y))
        {
            // counter += 1;
            LoopPatch(patch_size, offset,
                      inter_x, inter_y, cross_x, cross_y,
                      sigma2, dcos_x, dcos_y,
                      l_bound, b_bound, gx, gy,
                      projection_value + tid, image);
        }
    }
    // printf("count num is: %d\n", counter );
}

///
///backprojection
///
__global__ void BackComputeSlice(const float *x1, const float *y1, const float *z1,
                                const float *x2, const float *y2, const float *z2,
                                const unsigned int patch_size, const unsigned int offset,
                                const float slice_z,
                                const float l_bound, const float b_bound, const float sigma2,
                                const int gx, const int gy, const float inter_x, const float inter_y,
                                const float *projection_value, const int num_events, float *image)
{
    
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < num_events;
         tid += blockDim.x * gridDim.x)
    {
        float dcos_x = 0;
        float dcos_y = 0;
        float cross_x = 0;
        float cross_y = 0;
        if (CalculateCrossPoint(x1[tid], y1[tid], z1[tid],
                                x2[tid], y2[tid], z2[tid],
                                slice_z, dcos_x, dcos_y, cross_x, cross_y))
        {
            BackLoopPatch(patch_size, offset,
                      inter_x, inter_y, cross_x, cross_y,
                      sigma2, dcos_x, dcos_y,
                      l_bound, b_bound, gx, gy,
                      projection_value[tid], image);
        }
    }
}

void projection(const float *x1, const float *y1, const float *z1,
                const float *x2, const float *y2, const float *z2,
                float *projection_value,
                const int *grid, const float *center, const float *size,
                const float kernel_width,
                const float *image, const int num_events)
{
    // auto grid_flat = grid.flat<int>();
    // auto center_flat = center.flat<float>();
    // auto size_flat = size.flat<float>();
    // auto projection_value_flat = projection_value->flat<float>();
    // int num_events = projection_value->size();
    // std::cout << "here we stop0!" << std::endl;
    // std::cout << grid << std::endl;
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    // std::cout << gx << " " << gy << " " << gz << std::endl;
    float center_x = center_cpu[0], center_y = center_cpu[1], center_z = center_cpu[2]; // position of center
    float lx = size_cpu[0], ly = size_cpu[1], lz = size_cpu[2];                         // length of bounds
    unsigned int slice_mesh_num = gx * gy;                                              // number of meshes in a slice.

    float inter_x = lx / gx, inter_y = ly / gy, inter_z = lz / gz;  // intervals

    float l_bound = center_x - lx / 2, b_bound = center_y - ly / 2; // left and bottom bound of the slice.
    //float kernel_width = 3;                                         //this->app->get_kernel_width();
    //sigma2 indicate the bound of a gaussian kernel with the relationship: 3*sigma = kernel_width.
    // float kernel_width = 6*std::sqrt(sigma2);
    float sigma2 = kernel_width*kernel_width/36;
    int patch_size = std::ceil((std::sqrt(2)*kernel_width + lz / gz)/(lx / gx));
    // int patch_size = (kernel_width * 2 * std::sqrt(2) + (lz / gz)) / (lx / gx) + 1;
    // float dcos_x, dcos_y;

    for (unsigned int iSlice = 0; iSlice < gz; iSlice++)
    {
        int offset = iSlice * slice_mesh_num;
        float slice_z = center_z - (lz - inter_z) / 2 + iSlice * inter_z;
        ComputeSlice<<<32, 1024>>>(x1, y1, z1, x2, y2, z2,
                                  patch_size, offset, slice_z,
                                  l_bound, b_bound, sigma2,
                                  gx, gy, inter_x, inter_y,
                                  projection_value, num_events,
                                  image);
                                
    }
}

void backprojection(const float *x1, const float *y1, const float *z1,
                    const float *x2, const float *y2, const float *z2,
                    const float *projection_value,
                    const int *grid, const float *center, const float *size,
                    const float kernel_width,
                    float *image, const int num_events)
{
    // auto grid_flat = grid.flat<int>();
    // auto center_flat = center.flat<float>();
    // auto size_flat = size.flat<float>();
    // auto projection_value_flat = projection_value->flat<float>();
    // int num_events = projection_value->size();
    // std::cout << "here we stop0!" << std::endl;
    // std::cout << grid << std::endl;
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    // std::cout << gx << " " << gy << " " << gz << std::endl;
    float center_x = center_cpu[0], center_y = center_cpu[1], center_z = center_cpu[2]; // position of center
    float lx = size_cpu[0], ly = size_cpu[1], lz = size_cpu[2];                         // length of bounds
    unsigned int slice_mesh_num = gx * gy;                                              // number of meshes in a slice.

    float inter_x = lx / gx, inter_y = ly / gy, inter_z = lz / gz;  // intervals
    float l_bound = center_x - lx / 2, b_bound = center_y - ly / 2; // left and bottom bound of the slice.

    //sigma2 indicate the bound of a gaussian kernel with the relationship: 3*sigma = kernel_width.
    float sigma2 = kernel_width*kernel_width/36;
    int patch_size = std::ceil((std::sqrt(2)*kernel_width + lz / gz)/(lx / gx));
    // int patch_size = (kernel_width * 2 * std::sqrt(2) + (lz / gz)) / (lx / gx) + 1;

    std::cout<<"number of events: "<<num_events<<std::endl;

    for (unsigned int iSlice = 0; iSlice < gz; iSlice++)
    {
        int offset = iSlice * slice_mesh_num;
        float slice_z = center_z - (lz - inter_z) / 2.0 + iSlice * inter_z;
        BackComputeSlice<<<32, 1024>>>(x1, y1, z1, x2, y2, z2,
                                      patch_size, offset, slice_z,
                                      l_bound, b_bound, sigma2,
                                      gx, gy, inter_x, inter_y,
                                      projection_value, num_events,
                                      image);
    }
}

#endif

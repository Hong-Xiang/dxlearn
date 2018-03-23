#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"

using namespace tensorflow;
//using namespace BBSLMIRP;

REGISTER_OP("Projection")
    .Input("lors: float") 
    .Input("image: float")
    .Input("grid: int32")
    .Input("center: float")
    .Input("size: float")
    .Output("line_integral: float")
    .Attr("kernel_width: float")
    .Attr("model: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 1));
      return Status::OK();
    });

REGISTER_OP("Backprojection")
    .Input("image: float")
    .Input("grid: int32")
    .Input("center: float")
    .Input("size: float")
    .Input("lors: float") 
    .Input("line_integral: float")
    .Output("backpro_image: float")
    .Attr("kernel_width: float")
    .Attr("model: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      //set the size of backpro_image the same as the input image.
      c->set_output(0, c->input(0));
      return Status::OK();
    });


class Projection : public OpKernel
{
public:
  explicit Projection(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("model", &model));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width));
  }

  void Compute(OpKernelContext *context) override
  {
    // Grab the input tensor
    const Tensor &lors = context->input(0);
    const Tensor &image = context->input(1);
    const Tensor &grid = context->input(2);
    const Tensor &center = context->input(3);
    const Tensor &size = context->input(4);

    // Create an output tensor
    Tensor *projection_value = NULL;
  
    //debug
    //std::cout<<"lor_shape:"<<lors.shape().dim_size(1)<<std::endl;
    
    // define the shape of output tensors.
    TensorShape out_shape;
    out_shape.AddDim(lors.shape().dim_size(0));
    //debug
    //std::cout<<"out_shape dim:"<<out_shape.dims()<<std::endl;
    //std::cout<<"out_shape dim0:"<<out_shape.dim_size(0)<<std::endl;
    
    //assign the size of output projection values.
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &projection_value));
    //set teh initial projection_value to zero. 
    auto pv_flat = projection_value->flat<float>();
    for (int i = 0; i< pv_flat.size(); i++){
      pv_flat(i) = 0;
    }
    projection(lors, projection_value, grid, center, size, kernel_width, image);
  }

  void projection(const Tensor &events, Tensor* projection_value,
                      const Tensor &grid, const Tensor &center, const Tensor &size,
                      const float& kernel_width,
                      const Tensor& image)
  {
    auto grid_flat = grid.flat<int>();
    auto center_flat = center.flat<float>();
    auto size_flat = size.flat<float>();
    auto projection_value_flat = projection_value->flat<float>();
    unsigned int gx = grid_flat(0), gy = grid_flat(1), gz = grid_flat(2);                  //number of meshes
    float center_x = center_flat(0), center_y = center_flat(1), center_z = center_flat(2); // position of center
    float lx = size_flat(0), ly = size_flat(1), lz = size_flat(2);                         // length of bounds
    unsigned int slice_mesh_num = gx * gy;                                  // number of meshes in a slice.

    float inter_x = lx / gx, inter_y = lx / gy, inter_z = lz / gz;  // intervals
    float l_bound = center_x - lx / 2, b_bound = center_y - ly / 2; // left and bottom bound of the slice.
    //float kernel_width = 3;                                         //this->app->get_kernel_width();
    int patch_size = (kernel_width * 2 * std::sqrt(2) + (lz / gz)) / (lx / gx) + 1;
    //sigma2 indicate the bound of a gaussian kernel with the relationship: 3*sigma = kernel_width.
    float sigma2 = kernel_width * kernel_width / 9;
    float dcos_x, dcos_y;

    for (unsigned int iSlice = 0; iSlice < gz; iSlice++)
    {
      int offset = iSlice * slice_mesh_num;
      int slice_z = center_z - (lz - inter_z) / 2 + iSlice * inter_z;
      float cross_x, cross_y;
      unsigned int num_events = events.dim_size(0);
      //std::cout<<"events dims:"<<events.dim_size(0)<<", "<<events.dim_size(1)<<std::endl;
      for (unsigned int iEvent = 0; iEvent < num_events; iEvent++)
      {
        auto event = events.Slice(iEvent,iEvent+1);
        //debug
        //std::cout<<"event:"<<event.IsAligned()<<std::endl;
        if (CalculateCrossPoint(event, slice_z, dcos_x, dcos_y, cross_x, cross_y))
        {
          LoopPatch(patch_size, offset,
                    inter_x, inter_y, cross_x, cross_y,
                    sigma2, dcos_x, dcos_y,
                    l_bound, b_bound, projection_value_flat(iEvent), image);
        }
      }
    }
  }



  bool CalculateCrossPoint(const Tensor &event, const float& s_center,
                                         float &dcos_x, float &dcos_y,
                                         float &cross_x, float &cross_y)
  {
    // float pos1_x = 0, pos1_y = 0, pos1_z = 0;
    // float pos2_x = 0, pos2_y = 0, pos2_z = 0;
    auto event_flat = event.unaligned_flat<float>();
    float pos1_x = event_flat(0), pos1_y = event_flat(1), pos1_z = event_flat(2);
    float pos2_x = event_flat(3), pos2_y = event_flat(4), pos2_z = event_flat(5);
    float d0 = pos2_x -pos1_x, d1 = pos2_y - pos1_y, d2 = pos2_z - pos1_z;
    float ratio = (s_center - pos1_z) / d2;
    if (ratio < 0 || ratio > 1)
      return false;
    //the x and y value of cross point.
    cross_x = pos1_x + d0 * ratio;
    cross_y = pos1_y + d1 * ratio;
    //the len of lor.
    float dis = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    //direction cosine of lor.
    dcos_x = d0 / dis;
    dcos_y = d1 / dis;
    //printf("calculate points: %f \n",s_center);
    return true;
  }

  void LoopPatch(unsigned patch_size, unsigned int offset,
                 const float &inter_x, const float &inter_y,
                 const float &cross_x, const float &cross_y,
                 const float &sigma2,
                 const float &dcos_x, const float &dcos_y,
                 const float &l_bound, const float &b_bound,
                 float &projection_value, const Tensor& image)
  {
    //
    auto image_flat = image.flat<float>();
    int l0 = image.dim_size(0);
    int l1 = image.dim_size(1);
    //the start mesh of
    int index_x = (int)((cross_x - l_bound) / inter_x) - (int)(patch_size / 2);
    int index_y = (int)((cross_y - b_bound) / inter_y) - (int)(patch_size / 2);
    for (int j = 0; j < patch_size; j++)
    {
      for (int i = 0; i < patch_size; i++)
      {
        int index0 = index_x + i;
        int index1 = index_y + j;
        if (index0 < 0 || index1 < 0 || index0 >= l0 || index1 >= l1)
        {
          continue;
        }
        else
        {
          int index = index0 + index1 * l0;
          float value = 0;
          // compute the system matrix value.
          CalculateSMV(cross_x, cross_y,
                       inter_x * (index0 + 0.5) + l_bound, inter_y * (index1 + 0.5) + b_bound,
                       sigma2, dcos_x, dcos_y, value);
          projection_value += image_flat(offset + index) * value;
        }
      }
    }
  }
  //
  void CalculateSMV(const float &cross_x, const float &cross_y,
                    const float &mesh_x, const float &mesh_y,
                    const float &sigma2,
                    const float &dcos_x, const float &dcos_y,
                    float &value)
  {
    float delta_x = cross_x - mesh_x;
    float delta_y = cross_y - mesh_y;
    float r_cos = (delta_x * dcos_x + delta_y * dcos_y);
    float d2 = delta_x * delta_x + delta_y * delta_y - r_cos * r_cos;
    value = (d2 < 9 * sigma2) ? std::exp(-0.5 * d2 / sigma2) : 0;
  }

private:
  string model;
  float kernel_width;

};

class Backprojection : public OpKernel
{
public:
  explicit Backprojection(OpKernelConstruction *context) : OpKernel(context)
  {
    OP_REQUIRES_OK(context, context->GetAttr("model", &model));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width));
  }

  void Compute(OpKernelContext *context) override
  {

    // Grab the geometries of an image.
    const Tensor &image = context->input(0);
    const Tensor &grid = context->input(1);
    const Tensor &center = context->input(2);
    const Tensor &size = context->input(3);
    // Grab the input lors.
    const Tensor &events = context->input(4);
    //grab the input projection values(line integral)
    const Tensor &projection_value = context->input(5);

    // Create an output backprojected image
    Tensor *backpro_image = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(),
                                                     &backpro_image));
    //set the initial backprojection image value to zero.
    auto backpro_image_flat = backpro_image->flat<float>();
    for(int i = 0; i < backpro_image_flat.size(); ++i){
      backpro_image_flat(i) = 0;
    }
    backprojection(events, projection_value, grid, center, size, kernel_width, backpro_image);
  }

  void backprojection(const Tensor &events, const Tensor &projection_value,
                      const Tensor &grid, const Tensor &center, const Tensor &size,
                      const float& kernel_width,
                      Tensor* backpro_image)
  {
    auto grid_flat = grid.flat<int>();
    auto center_flat = center.flat<float>();
    auto size_flat = size.flat<float>();
    auto projection_value_flat = projection_value.flat<float>(); 
    unsigned int gx = grid_flat(0), gy = grid_flat(1), gz = grid_flat(2);                  //number of meshes
    float center_x = center_flat(0), center_y = center_flat(1), center_z = center_flat(2); // position of center
    float lx = size_flat(0), ly = size_flat(1), lz = size_flat(2);                         // length of bounds
    unsigned int slice_mesh_num = gx * gy;                                  // number of meshes in a slice.

    float inter_x = lx / gx, inter_y = lx / gy, inter_z = lz / gz;  // intervals
    float l_bound = center_x - lx / 2, b_bound = center_y - ly / 2; // left and bottom bound of the slice.
    //float kernel_width = 3;                                         //this->app->get_kernel_width();
    int patch_size = (kernel_width * 2 * std::sqrt(2) + (lz / gz)) / (lx / gx) + 1;
    //sigma2 indicate the bound of a gaussian kernel with the relationship: 3*sigma = kernel_width.
    float sigma2 = kernel_width * kernel_width / 9;
    float dcos_x, dcos_y;

    for (unsigned int iSlice = 0; iSlice < gz; iSlice++)
    {
      int offset = iSlice * slice_mesh_num;
      int slice_z = center_z - (lz - inter_z) / 2 + iSlice * inter_z;
      float cross_x, cross_y;
      unsigned int num_events = events.dim_size(0);
      // std::cout<<"events dims:"<<events.dim_size(0)<<", "<<events.dim_size(1)<<std::endl;
      for (unsigned int iEvent = 0; iEvent < num_events; iEvent++)
      {
        auto event = events.Slice(iEvent,iEvent+1);
        // std::cout<<"event:"<<event.IsAligned()<<std::endl;
        if (CalculateCrossPoint(event, slice_z, dcos_x, dcos_y, cross_x, cross_y))
        {
          LoopPatch(patch_size, offset,
                    inter_x, inter_y, cross_x, cross_y,
                    sigma2, dcos_x, dcos_y,
                    l_bound, b_bound, projection_value_flat(iEvent), backpro_image);
        }
      }
    }
  }
  bool CalculateCrossPoint(const Tensor &event, const float& s_center,
                                         float &dcos_x, float &dcos_y,
                                         float &cross_x, float &cross_y)
  {
    // float pos1_x = 0, pos1_y = 0, pos1_z = 0;
    // float pos2_x = 0, pos2_y = 0, pos2_z = 0;
    auto event_flat = event.unaligned_flat<float>();
    float pos1_x = event_flat(0), pos1_y = event_flat(1), pos1_z = event_flat(2);
    float pos2_x = event_flat(3), pos2_y = event_flat(4), pos2_z = event_flat(5);
    float d0 = pos2_x -pos1_x, d1 = pos2_y - pos1_y, d2 = pos2_z - pos1_z;
    float ratio = (s_center - pos1_z) / d2;
    if (ratio < 0 || ratio > 1)
      return false;
    //the x and y value of cross point.
    cross_x = pos1_x + d0 * ratio;
    cross_y = pos1_y + d1 * ratio;
    //the len of lor.
    float dis = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    //direction cosine of lor.
    dcos_x = d0 / dis;
    dcos_y = d1 / dis;
    //printf("calculate points: %f \n",s_center);
    return true;
  }

  void LoopPatch(unsigned patch_size, unsigned int offset,
                 const float &inter_x, const float &inter_y,
                 const float &cross_x, const float &cross_y,
                 const float &sigma2,
                 const float &dcos_x, const float &dcos_y,
                 const float &l_bound, const float &b_bound,
                 const float &projection_value, Tensor* backpro_image)
  {
    //
    auto image_flat = backpro_image->flat<float>();
    int l0 = backpro_image->dim_size(0);
    int l1 = backpro_image->dim_size(1);
    // std::cout<<"l0:"<<l0<<" l1:"<<l1<<std::endl;
    //the start mesh of
    int index_x = (int)((cross_x - l_bound) / inter_x) - (int)(patch_size / 2);
    int index_y = (int)((cross_y - b_bound) / inter_y) - (int)(patch_size / 2);
    for (int j = 0; j < patch_size; j++)
    {
      for (int i = 0; i < patch_size; i++)
      {
        int index0 = index_x + i;
        int index1 = index_y + j;
        if (index0 < 0 || index1 < 0 || index0 >= l0 || index1 >= l1)
        {
          continue;
        }
        else
        {
          int index = index0 + index1 * l0;
          float value = 0;
          // compute the system matrix value.
          CalculateSMV(cross_x, cross_y,
                       inter_x * (index0 + 0.5) + l_bound, inter_y * (index1 + 0.5) + b_bound,
                       sigma2, dcos_x, dcos_y, value);
          if (projection_value < 1e-5)
            image_flat(offset + index) += 0;
          else
            image_flat(offset + index) += value / projection_value;
            //debug
            // std::cout<<"projection:"<<projection_value<<std::endl;
            // std::cout<<"image_flat:"<<image_flat(offset + index)<<std::endl;
        }
      }
    }
  }
  //
  void CalculateSMV(const float &cross_x, const float &cross_y,
                    const float &mesh_x, const float &mesh_y,
                    const float &sigma2,
                    const float &dcos_x, const float &dcos_y,
                    float &value)
  {
    float delta_x = cross_x - mesh_x;
    float delta_y = cross_y - mesh_y;
    float r_cos = (delta_x * dcos_x + delta_y * dcos_y);
    float d2 = delta_x * delta_x + delta_y * delta_y - r_cos * r_cos;
    value = (d2 < 9 * sigma2) ? std::exp(-0.5 * d2 / sigma2) : 0;
  }
  
  private : 
    string model;
    float kernel_width;
};

#define REGISTER_CPU_KERNEL(name, op) \
  REGISTER_KERNEL_BUILDER(            \
      Name(name).Device(DEVICE_CPU), op)

REGISTER_CPU_KERNEL("Projection", Projection);
REGISTER_CPU_KERNEL("Backprojection", Backprojection);

#undef REGISTER_CPU_KERNEL
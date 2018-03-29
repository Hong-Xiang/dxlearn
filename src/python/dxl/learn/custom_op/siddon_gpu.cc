#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
const float realmin = 1.0e-13; //DBL_MIN
const float realmax = 1.0e13;  //DBL_MAX
const float SPD = 3e10;		   //light speed (cm/s)
const int EVENT_SIZE = 7;
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"

using namespace tensorflow;
//using namespace BBSLMIRP;

#include <vector>
struct TOFInfo
{
	const float time_resolution;
	const float tof_bin;
};
struct EventsInfo
{
	EventsInfo(int nb_events, int event_size)
		: nb_events(nb_events), event_size(event_size) {}
	const long long nb_events;
	const int event_size;
};
struct ImageInfo
{
	const int grid[3];
	const float center[3];
	const float size[3];
};

REGISTER_OP("ProjectionGpu")
	.Input("lors: float")
	.Input("image: float")
	.Output("result: float")
	.Attr("grid: list(int)")
	.Attr("origin: list(float)")
	.Attr("size: list(float)")
	.Attr("tof_bin: float")
	.Attr("time_resolution: float")
	.Attr("model: string")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 1));
		return Status::OK();
	});

void ProjectionKernelLauncher(const float *lor,
							  const EventsInfo events_info,
							  const float *image,
							  const ImageInfo image_info,
							  const TOFInfo tof_info,
							  float *result);

class Projection : public OpKernel
{
  public:
	explicit Projection(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("grid", &grid));
		OP_REQUIRES_OK(context, context->GetAttr("origin", &center));
		OP_REQUIRES_OK(context, context->GetAttr("size", &size));
		OP_REQUIRES_OK(context, context->GetAttr("tof_bin", &tof_bin));
		OP_REQUIRES_OK(context, context->GetAttr("time_resolution", &time_resolution));
		OP_REQUIRES_OK(context, context->GetAttr("model", &model));
	}

	void Compute(OpKernelContext *context) override
	{
		const Tensor &lors = context->input(0);
		const Tensor &image = context->input(1);

		// TODO: Better way of constructions?
		TensorShape out_shape;
		out_shape.AddDim(lors.shape().dim_size(0));
		Tensor *result = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
														 &result));
		auto result_flat = result->flat<float>();
		cudaMemset(result_flat.data(), 0, sizeof(float) * result_flat.size());
		ProjectionKernelLauncher(lors.flat<float>().data(),
								 EventsInfo{lors.dim_size(0), EVENT_SIZE},
								 image.flat<float>().data(),
								 ImageInfo{{grid[0], grid[1], grid[2]},
										   {center[0], center[1], center[2]},
										   {size[0], size[1], size[2]}},
								 TOFInfo{time_resolution, tof_bin},
								 result_flat.data());
	}

  private:
	std::vector<int> grid;
	std::vector<float> center;
	std::vector<float> size;
	float tof_bin;
	float time_resolution;
	string model;
};

REGISTER_OP("BackprojectionGpu")
	.Input("lors:float")
	.Input("lor_values: float")
	.Input("image: float")
	.Output("result: float")
	.Attr("grid: list(int)")
	.Attr("origin: list(float)")
	.Attr("size: list(float)")
	.Attr("tof_bin: float")
	.Attr("time_resolution: float")
	.Attr("model: string")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
			c->set_output(0, c->input(2));
			return Status::OK();
		});

void BackProjectionKernelLauncher(const float *lor,
								  const float *lor_values,
								  const EventsInfo events_info,
								  const ImageInfo image_info,
								  const TOFInfo tof_info,
								  float *result);

class Backprojection : public OpKernel
{
  public:
	explicit Backprojection(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("grid", &grid));
		OP_REQUIRES_OK(context, context->GetAttr("origin", &center));
		OP_REQUIRES_OK(context, context->GetAttr("size", &size));
		OP_REQUIRES_OK(context, context->GetAttr("tof_bin", &tof_bin));
		OP_REQUIRES_OK(context, context->GetAttr("time_resolution", &time_resolution));
		OP_REQUIRES_OK(context, context->GetAttr("model", &model));
	}

	void Compute(OpKernelContext *context) override
	{

		const Tensor &lors = context->input(0);		  // Grab the input lors.
		const Tensor &lor_values = context->input(1); //grab the input lor values
		const Tensor &image = context->input(2);

		Tensor *result = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(), &result));
		auto result_flat = result->flat<float>();
		cudaMemset(result_flat.data(), 0, sizeof(float) * result_flat.size());
		BackProjectionKernelLauncher(lors.flat<float>().data(),
									 lor_values.flat<float>().data(),
									 EventsInfo{lors.dim_size(0), EVENT_SIZE},
									 ImageInfo{{grid[0], grid[1], grid[2]},
											   {center[0], center[1], center[2]},
											   {size[0], size[1], size[2]}},
									 TOFInfo{time_resolution, tof_bin},
									 result_flat.data());
	}

  private:
	std::vector<int> grid;
	std::vector<float> center;
	std::vector<float> size;
	float tof_bin;
	float time_resolution;
	string model;
};

REGISTER_KERNEL_BUILDER(Name("ProjectionGpu").Device(DEVICE_GPU), Projection)
REGISTER_KERNEL_BUILDER(Name("BackprojectionGpu").Device(DEVICE_GPU), Backprojection)
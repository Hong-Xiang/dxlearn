#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow; // NOLINT(build/namespaces)

REGISTER_OP("Projection")
    .Input("input: float32")
    .Output("projection: float32");

void ProjectionKernelLauncher(const float *in, const int N, float *out);

class ProjectionGpuOp : public OpKernel
{
  public:
    explicit ProjectionGpuOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<float>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        // Call the cuda kernel launcher
        ProjectionKernelLauncher(input.data(), N, output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("Projection").Device(DEVICE_GPU), ProjectionGpuOp);
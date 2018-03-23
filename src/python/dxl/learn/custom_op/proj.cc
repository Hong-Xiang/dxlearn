#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"

using namespace tensorflow;
//using namespace BBSLMIRP;

REGISTER_OP("Projection")
    .Input("to_zero: float32")
    .Output("zeroed: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("BackProjection")
    .Input("to_one: int32")
    .Input("to_f: float32")
    .Output("oneed: int32")
    .Output("fed: float32")
    .Attr("model: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status::OK();
    });

class Projection : public OpKernel
{
  public:
    explicit Projection(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<float>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i < N; i++)
        {
            output_flat(i) = 0;
        }

        // Preserve the first input value if possible.
        if (N > 0)
            output_flat(0) = input(0);
    }
};

class BackProjection : public OpKernel
{
  public:
    explicit BackProjection(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("model", &model));
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &events = context->input(0);
        auto input = events.flat<int32>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, events.shape(),
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<int32>();

        // // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i < N; i++)
        {
            output_flat(i) = 1;
        }

        // Preserve the first input value if possible.
        if (N > 0)
            output_flat(0) = input(0);

        const Tensor &input_tensor_f = context->input(1);
        auto input1 = input_tensor_f.flat<float>();

        Tensor *output_tensor_f = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor_f.shape(),
                                                         &output_tensor_f));

        auto output_flat1 = output_tensor_f->flat<float>();

        const int N1 = input1.size();
        int val = 0;
        if (model == "siddon")
        {
            val = 100;
        }
        else
        {
            val = 1;
        }
        for (int i = 0; i < N1; i++)
        {
            output_flat1(i) = input1(i) + val;
        }
    }

  private:
    string model;
};
REGISTER_KERNEL_BUILDER(Name("Projection").Device(DEVICE_CPU), Projection);
REGISTER_KERNEL_BUILDER(Name("BackProjection").Device(DEVICE_CPU), BackProjection);
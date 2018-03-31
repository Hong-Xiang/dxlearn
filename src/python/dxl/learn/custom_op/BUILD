
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
tf_custom_op_library(
    name = "proj.so",
    srcs = ["proj.cc"],
)
tf_custom_op_library(
    name = "pet.so",
    srcs = ["pet.cc"]
)
tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)
tf_custom_op_library(
    name = "projection_gpu.so",
    srcs = ["projection_gpu.cc"],
    gpu_srcs = ["projection_gpu.cu.cc"],
)
tf_custom_op_library(
    name = "pet_gpu.so",
    srcs = ["pet_gpu.cc"],
    gpu_srcs = ["pet_gpu.cu.cc"],
)

tf_custom_op_library(
    name = "tof_tor.so",
    srcs = ["tof_tor.cc"],
    gpu_srcs = ["tof_tor.cu.cc"]
)

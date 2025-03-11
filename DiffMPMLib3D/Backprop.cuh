#pragma once
#ifndef BACKPROP_CUH
#define BACKPROP_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Elasticity.h"
#include "MaterialPoint.h"
#include "GridNode.h"


namespace DiffMPMLib3D {
    __global__ void printout();
    __global__ void P_op_2_kernel(MaterialPoint* d_pc_points, MaterialPoint* d_pc_prev_points, float dt, int sz);
    __global__ void G2P_kernel(MaterialPoint* d_pc_prev_points, GridNode* d_nodes, float dx, int dim_x, int dim_y, int dim_z, int sz);
    __global__ void G_op_kernel(GridNode* d_nodes, int dim_x, int dim_y, int dim_z);
    __global__ void P2G_kernel(MaterialPoint* d_pc_points, MaterialPoint* d_pc_prev_points, GridNode* d_nodes,
        float dx, float dt, float drag, int dim_x, int dim_y, int dim_z, int sz);

    __device__ void QueryPoint_CubicBSpline_GPU(const Vec3& point, const Vec3& min_point, float dx, int dim_x, int dim_y, int dim_z, GridNode* gridNodes, GridNode** resultNodes, int* numResults);
    __global__ void QueryPoint_CubicBSpline_kernel(const Vec3 point, const Vec3 min_point, float dx, int dim_x, int dim_y, int dim_z, GridNode* nodes, GridNode** result_nodes, int* result_indices, int range);
    __global__ void d2_FCE_psi_dF2_mult_by_dF_kernel(const DiffMPMLib3D::Mat3* F, float lam, float mu, const Mat3* dF, Mat3* result);
    __global__ void d_JFit_dF_FD_kernel(const Mat3* F, Tensor3x3x3x3* result, int numElements);
}

#endif // BACKPROP_CUH

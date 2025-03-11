#include "Backprop.cuh"
#include "device_atomic_functions.h"

__device__ inline float CubicBSplineGPU(float x) {
    x = abs(x);
    if (0.0 <= x && x < 1.0) {
        return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
    }
    else if (1.0 <= x && x < 2.0) {
        return (2.0f - x) * (2.0f - x) * (2.0f - x) / 6.0f;
    }
    else {
        return 0.0f;
    }
}

__device__ inline float CubicBSplineSlopeGPU(float x) {
    float absx = abs(x);
    if (0.0 <= absx && absx < 1.0) {
        return 1.5f * x * absx - 2.0f * x;
    }
    else if (1.0 <= absx && absx < 2.0) {
        return -x * absx / 2.0f + 2.0f * x - 2.0f * x / absx;
    }
    else {
        return 0.0f;
    }
}

__device__ float dot_product_GPU(const DiffMPMLib3D::Mat3& A, const DiffMPMLib3D::Mat3& B) {
    return (A.array() * B.array()).sum();
}


__device__ void DiffMPMLib3D::QueryPoint_CubicBSpline_GPU(const Vec3& point, const Vec3& min_point, float dx, int dim_x, int dim_y, int dim_z, GridNode* gridNodes, GridNode** resultNodes, int* numResults) {
    Vec3 relative_point = point - min_point;
    int bot_left_index[3];

    for (int i = 0; i < 3; i++) {
        bot_left_index[i] = static_cast<int>((relative_point[i] / dx) - 1.0);
    }

    int range = 4;  // checking a 4x4x4 cube around the bottom left index
    *numResults = 0;

    for (int idx = 0; idx < range * range * range; idx++) {
        int k = idx % range;
        int j = (idx / range) % range;
        int i = idx / (range * range);

        int x = bot_left_index[0] + i;
        int y = bot_left_index[1] + j;
        int z = bot_left_index[2] + k;

        if (x >= 0 && x < dim_x && y >= 0 && y < dim_y && z >= 0 && z < dim_z) {
            resultNodes[*numResults] = &gridNodes[x * dim_y * dim_z + y * dim_z + z];
            (*numResults)++;
        }
    }
}

__global__ void DiffMPMLib3D::QueryPoint_CubicBSpline_kernel(const Vec3 point, 
    const Vec3 min_point, float dx, int dim_x, int dim_y, int dim_z, GridNode* nodes, GridNode** result_nodes, int* result_indices, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = range * range * range;

    if (idx >= total_elements) return;

    Vec3 relative_point = point - min_point;
    int bot_left_index[3];

    for (int i = 0; i < 3; i++) {
        bot_left_index[i] = static_cast<int>((relative_point[i] / dx) - 1.f);
    }

    int k = idx % range;
    int j = (idx / range) % range;
    int i = idx / (range * range);

    int x = bot_left_index[0] + i;
    int y = bot_left_index[1] + j;
    int z = bot_left_index[2] + k;

    if (x >= 0 && x < dim_x && y >= 0 && y < dim_y && z >= 0 && z < dim_z) {
        int index = x + y * dim_x + z * dim_x * dim_y;
        result_nodes[idx] = &nodes[index];
        if (result_indices != nullptr) {
            result_indices[idx * 3 + 0] = x;
            result_indices[idx * 3 + 1] = y;
            result_indices[idx * 3 + 2] = z;
        }
    }
    else {
        result_nodes[idx] = nullptr;
        if (result_indices != nullptr) {
            result_indices[idx * 3 + 0] = -1;
            result_indices[idx * 3 + 1] = -1;
            result_indices[idx * 3 + 2] = -1;
        }
    }
}


__device__ void gramSchmidtOrthogonalization(const DiffMPMLib3D::Mat3& F, DiffMPMLib3D::Mat3& R) {
    DiffMPMLib3D::Vec3 u1 = F.col(0);
    float norm_u1 = u1.norm();
    R.col(0) = u1 / norm_u1;

    DiffMPMLib3D::Vec3 u2 = F.col(1);
    u2 -= u2.dot(R.col(0)) * R.col(0);
    float norm_u2 = u2.norm();
    R.col(1) = u2 / norm_u2;

    DiffMPMLib3D::Vec3 u3 = F.col(2);
    u3 -= u3.dot(R.col(0)) * R.col(0);
    u3 -= u3.dot(R.col(1)) * R.col(1);
    float norm_u3 = u3.norm();
    R.col(2) = u3 / norm_u3;
}

__device__ void polarDecomposition_GPU(const DiffMPMLib3D::Mat3& F, DiffMPMLib3D::Mat3& R, DiffMPMLib3D::Mat3& S) {
    gramSchmidtOrthogonalization(F, R);
    S = R.transpose() * F;
}

__global__ void DiffMPMLib3D::printout() {
    printf("Hello? I'm GPU!\n");
}

__global__ void DiffMPMLib3D::d_JFit_dF_FD_kernel(const Mat3* F, Tensor3x3x3x3* result, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) return;

    Mat3 temp_F = F[idx];
    float delta = 1e-9;
    Tensor3x3x3x3 ret;

    auto CalcJFit = [](const Mat3& _F)
        {
            float J = _F.determinant();
            Mat3 Fit = _F.inverse().transpose();
            Mat3 ret = J * Fit;
            return ret;
        };


    for (int index = 0; index < 81; index++) {
        int i = (index / 9) / 3;
        int j = (index / 9) % 3;
        int a = (index % 9) / 3;
        int b = (index % 9) % 3;

        float originalValue = temp_F(i, j);

        temp_F(i, j) = originalValue + delta;
        Mat3 JFit_forward = CalcJFit(temp_F);

        temp_F(i, j) = originalValue - delta;
        Mat3 JFit_backward = CalcJFit(temp_F);

        temp_F(i, j) = originalValue;
        float l1 = JFit_forward(a, b);
        float l2 = JFit_backward(a, b);
        //ret[a][b](i, j) = (l1 - l2) / (2.f * delta); //ERROR
    }

    result[idx] = ret;
}

__global__ void DiffMPMLib3D::d2_FCE_psi_dF2_mult_by_dF_kernel(const DiffMPMLib3D::Mat3* F, float lam, float mu, const Mat3* dF, Mat3* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Mat3 R, S;
    polarDecomposition_GPU(F[idx], R, S);
    float J = F[idx].determinant();
    Mat3 Finv = F[idx].inverse();
    Mat3 Fit = Finv.transpose();

    Mat3 ret = Mat3::Zero();

    // 2 mu dF 
    ret += 2.f * mu * dF[idx];

    // + lam J F^-T (J F^-T : dF)
    ret += lam * J * Fit * dot_product_GPU(J * Fit, dF[idx]);

    Mat3 term = Mat3::Zero();

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            float local_contribution = 0.0f;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    // Tensor dJFit_dF 계산이 필요하지만 여기서는 생략
                    // local_contribution += dJFit_dF[a][b](i, j) * dF[idx](i, j);
                }
            }
            term(a, b) += local_contribution;
        }
    }

    ret += lam * (J - 1.f) * term;

    float s00, s01, s02, s11, s12, s22;

    s00 = S(0, 0);
    s01 = S(0, 1);
    s02 = S(0, 2);
    s11 = S(1, 1);
    s12 = S(1, 2);
    s22 = S(2, 2);
    Mat3 A;
    A(0, 0) = s02;
    A(0, 1) = s12;
    A(0, 2) = -(s00 + s11);
    A(1, 0) = -s01;
    A(1, 1) = s00 + s22;
    A(1, 2) = -s12;
    A(2, 0) = -(s11 + s22);
    A(2, 1) = s01;
    A(2, 2) = s02;

    Mat3 temp = R.transpose() * dF[idx] - dF[idx].transpose() * R;
    Vec3 b(temp(0, 1), temp(0, 2), temp(1, 2));

    // Let x = R.T * dR
    // solve Ax = b for x using Eigen solver
    //Vec3 x = A.colPivHouseholderQr().solve(b);

    //// dR = R(R.T * dR)
    //Mat3 dR;
    //dR << 0.f, -x(2), x(1),
    //    x(2), 0.f, -x(0),
    //    -x(1), x(0), 0.f;
    //dR = R * dR;

    //ret -= 2.f * mu * dR;

    //result[idx] = ret;
}


__global__ void DiffMPMLib3D::P_op_2_kernel(MaterialPoint* d_pc_points, MaterialPoint* d_pc_prev_points, float dt, int sz) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < sz) {
        const MaterialPoint& mp = d_pc_points[p];
        MaterialPoint& mp_prev = d_pc_prev_points[p];

        mp_prev.dLdv_next = dt * mp.dLdx + mp.dLdv;
        mp_prev.dLdC_next = dt * mp.dLdF * (mp_prev.F + mp_prev.dFc).transpose() + mp.dLdC;
    }
}


__global__ void DiffMPMLib3D::G2P_kernel(MaterialPoint* d_pc_prev_points, GridNode* d_nodes, float dx, int dim_x, int dim_y, int dim_z, int sz) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < sz) {
        const MaterialPoint& mp_prev = d_pc_prev_points[p];

        int range = 4;
        int half_range = range / 2;
        int base_x = static_cast<int>((mp_prev.x[0] - half_range) / dx);
        int base_y = static_cast<int>((mp_prev.x[1] - half_range) / dx);
        int base_z = static_cast<int>((mp_prev.x[2] - half_range) / dx);

        for (int i = 0; i < range; ++i) {
            for (int j = 0; j < range; ++j) {
                for (int k = 0; k < range; ++k) {
                    int x = base_x + i;
                    int y = base_y + j;
                    int z = base_z + k;

                    if (x >= 0 && x < dim_x && y >= 0 && y < dim_y && z >= 0 && z < dim_z) {
                        int idx = x * dim_y * dim_z + y * dim_z + z;
                        GridNode& node = d_nodes[idx];
                        Vec3 xg = node.x;
                        Vec3 xp = mp_prev.x;
                        Vec3 dgp = xg - xp;
                        float wgp = CubicBSplineGPU(dgp[0] / dx) * CubicBSplineGPU(dgp[1] / dx) * CubicBSplineGPU(dgp[2] / dx);

                        atomicAdd(&node.dLdv[0], mp_prev.dLdv_next[0] * wgp + 3.f / (dx * dx) * wgp * mp_prev.dLdC_next(0,0) * dgp[0]);
                        atomicAdd(&node.dLdv[1], mp_prev.dLdv_next[1] * wgp + 3.f / (dx * dx) * wgp * mp_prev.dLdC_next(1,1) * dgp[1]);
                        atomicAdd(&node.dLdv[2], mp_prev.dLdv_next[2] * wgp + 3.f / (dx * dx) * wgp * mp_prev.dLdC_next(2,2) * dgp[2]);
                    }
                }
            }
        }
    }
}

__global__ void DiffMPMLib3D::G_op_kernel(GridNode * d_nodes, int dim_x, int dim_y, int dim_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / (dim_y * dim_z); // i index
    int j = (idx / dim_z) % dim_y; // j index
    int k = idx % dim_z;           // k index

   /* if (i < dim_x && j < dim_y && k < dim_z) {
        GridNode& node = d_nodes[idx];
        if (fabs(node.m) > std::numeric_limits<float>::epsilon()) {
            node.dLdp = node.dLdv / node.m;
            node.dLdm = -1.f / node.m * node.v.dot(node.dLdv);
        }
    }*/
}

__global__ void DiffMPMLib3D::P2G_kernel(MaterialPoint* d_pc_points, MaterialPoint* d_pc_prev_points, GridNode* d_nodes, float dx, float dt, float drag, int dim_x, int dim_y, int dim_z, int sz) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < sz) {
        const MaterialPoint& mp = d_pc_points[p];
        MaterialPoint& mp_prev = d_pc_prev_points[p];

        const Vec3& xp = mp_prev.x;
        const Mat3 F_transpose = (mp_prev.F + mp_prev.dFc).transpose();

        int range = 4;
        int half_range = range / 2;
        int base_x = static_cast<int>((xp[0] - half_range) / dx);
        int base_y = static_cast<int>((xp[1] - half_range) / dx);
        int base_z = static_cast<int>((xp[2] - half_range) / dx);

        for (int i = 0; i < range; ++i) {
            for (int j = 0; j < range; ++j) {
                for (int k = 0; k < range; ++k) {
                    int x = base_x + i;
                    int y = base_y + j;
                    int z = base_z + k;

                    if (x >= 0 && x < dim_x && y >= 0 && y < dim_y && z >= 0 && z < dim_z) {
                        int idx = x * dim_y * dim_z + y * dim_z + z;
                        GridNode& node = d_nodes[idx];

                        const Vec3& xg = node.x;
                        Vec3 dgp = xg - xp;

                        float dgp0_dx = dgp[0] / dx;
                        float dgp1_dx = dgp[1] / dx;
                        float dgp2_dx = dgp[2] / dx;

                        float bspline_dgp0 = CubicBSplineGPU(dgp0_dx);
                        float bspline_dgp1 = CubicBSplineGPU(dgp1_dx);
                        float bspline_dgp2 = CubicBSplineGPU(dgp2_dx);

                        float bspline_slope_dgp0 = CubicBSplineSlopeGPU(dgp0_dx);
                        float bspline_slope_dgp1 = CubicBSplineSlopeGPU(dgp1_dx);
                        float bspline_slope_dgp2 = CubicBSplineSlopeGPU(dgp2_dx);

                        float wgp = bspline_dgp0 * bspline_dgp1 * bspline_dgp2;

                        Vec3 wgpGrad = -1.f / dx * Vec3(
                            bspline_slope_dgp0 * bspline_dgp1 * bspline_dgp2,
                            bspline_dgp0 * bspline_slope_dgp1 * bspline_dgp2,
                            bspline_dgp0 * bspline_dgp1 * bspline_slope_dgp2
                        );

                        Mat3 G = -3.f / (dx * dx) * dt * mp_prev.vol * mp_prev.P * F_transpose + mp_prev.m * mp_prev.C;

                        atomicAdd(&mp_prev.dLdP(0, 0), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * node.dLdp[0] * (F_transpose * dgp).transpose()(0, 0));
                        atomicAdd(&mp_prev.dLdP(1, 1), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * node.dLdp[1] * (F_transpose * dgp).transpose()(1, 1));
                        atomicAdd(&mp_prev.dLdP(2, 2), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * node.dLdp[2] * (F_transpose * dgp).transpose()(2, 2));

                        atomicAdd(&mp_prev.dLdF(0, 0), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * dgp[0] * (mp_prev.P.transpose() * node.dLdp).transpose()(0, 0));
                        atomicAdd(&mp_prev.dLdF(1, 1), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * dgp[1] * (mp_prev.P.transpose() * node.dLdp).transpose()(1, 1));
                        atomicAdd(&mp_prev.dLdF(2, 2), -wgp * 3.f / (dx * dx) * dt * mp_prev.vol * dgp[2] * (mp_prev.P.transpose() * node.dLdp).transpose()(2, 2));

                        atomicAdd(&mp_prev.dLdC(0, 0), wgp * mp_prev.m * node.dLdp[0] * dgp[0]);
                        atomicAdd(&mp_prev.dLdC(0, 1), wgp * mp_prev.m * node.dLdp[0] * dgp[1]);
                        atomicAdd(&mp_prev.dLdC(0, 2), wgp * mp_prev.m * node.dLdp[0] * dgp[2]);
                        atomicAdd(&mp_prev.dLdC(1, 0), wgp * mp_prev.m * node.dLdp[1] * dgp[0]);
                        atomicAdd(&mp_prev.dLdC(1, 1), wgp * mp_prev.m * node.dLdp[1] * dgp[1]);
                        atomicAdd(&mp_prev.dLdC(1, 2), wgp * mp_prev.m * node.dLdp[1] * dgp[2]);
                        atomicAdd(&mp_prev.dLdC(2, 0), wgp * mp_prev.m * node.dLdp[2] * dgp[0]);
                        atomicAdd(&mp_prev.dLdC(2, 1), wgp * mp_prev.m * node.dLdp[2] * dgp[1]);
                        atomicAdd(&mp_prev.dLdC(2, 2), wgp * mp_prev.m * node.dLdp[2] * dgp[2]);

                        atomicAdd(&mp_prev.dLdx[0], mp_prev.m * node.dLdm * wgpGrad[0]);
                        atomicAdd(&mp_prev.dLdx[1], mp_prev.m * node.dLdm * wgpGrad[1]);
                        atomicAdd(&mp_prev.dLdx[2], mp_prev.m * node.dLdm * wgpGrad[2]);

                        Vec3 temp = dot_product_GPU(mp_prev.dLdC_next, node.v * dgp.transpose()) * wgpGrad;
                        temp -= wgp * mp_prev.dLdC_next.transpose() * node.v;

                        atomicAdd(&mp_prev.dLdx[0], 3.f / (dx * dx) * temp[0]);
                        atomicAdd(&mp_prev.dLdx[1], 3.f / (dx * dx) * temp[1]);
                        atomicAdd(&mp_prev.dLdx[2], 3.f / (dx * dx) * temp[2]);

                        atomicAdd(&mp_prev.dLdv[0], wgp * mp_prev.m * (1.f - dt * drag) * node.dLdp[0]);
                        atomicAdd(&mp_prev.dLdv[1], wgp * mp_prev.m * (1.f - dt * drag) * node.dLdp[1]);
                        atomicAdd(&mp_prev.dLdv[2], wgp * mp_prev.m * (1.f - dt * drag) * node.dLdp[2]);
                    }
                }
            }
        }

        // Back prop P_op_1
        mp_prev.dLdF += (Mat3::Identity() + dt * mp.C).transpose() * mp.dLdF;
        /*mp_prev.dLdF += d2_FCE_psi_dF2_mult_by_dF(mp_prev.F + mp_prev.dFc, mp_prev.lam, mp_prev.mu, mp_prev.dLdP);*/
        mp_prev.dLdx += mp.dLdx;
    }
}
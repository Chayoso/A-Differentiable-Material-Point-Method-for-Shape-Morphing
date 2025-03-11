#include "pch.h"
#include "BackPropagation.h"
#include "Elasticity.h"
#include "Interpolation.h"

void DiffMPMLib3D::Back_Timestep(CompGraphLayer& layer_nplus1, CompGraphLayer& layer_n, float drag, float dt, float smoothing_factor)
{
    const PointCloud& pc = *layer_nplus1.point_cloud;
    PointCloud& pc_prev = *layer_n.point_cloud;
    Grid& grid = *layer_n.grid;
    float dx = grid.dx;
    int sz = pc.points.size();

    // Back prop P_op_2
#pragma omp parallel for //schedule(guided, 8)
    for (int p = 0; p < sz; p++) {
        const MaterialPoint& mp = pc.points[p];
        MaterialPoint& mp_prev = pc_prev.points[p];

        mp_prev.dLdF = (Mat3::Identity() + dt * mp.C).transpose() * mp.dLdF;
        mp_prev.dLdF = mp_prev.dLdF * (1.0f - smoothing_factor) + mp.dLdF * smoothing_factor;

        mp_prev.dLdv_next = dt * mp.dLdx + mp.dLdv;
        mp_prev.dLdC_next = dt * mp.dLdF * (mp_prev.F + mp_prev.dFc).transpose() + mp.dLdC;
    }

    // Back prop G2P
#pragma omp parallel for //schedule(guided, 8)
    for (int p = 0; p < sz; p++) {
        const MaterialPoint& mp_prev = pc_prev.points[p];

        auto nodes = grid.QueryPoint_CubicBSpline(mp_prev.x);

        for (int i = 0; i < nodes.size(); i++) {
            GridNode& node = nodes[i];

            Vec3 xg = node.x;
            Vec3 xp = mp_prev.x;
            Vec3 dgp = xg - xp;
            float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

            node.dLdv += mp_prev.dLdv_next * wgp + 3.f / (dx * dx) * wgp * mp_prev.dLdC_next * dgp;
        }
    }

    // Back prop G_opschedule(guided, 8)
#pragma omp parallel for //schedule(guided, 8)
    for (int idx = 0; idx < grid.dim_x * grid.dim_y * grid.dim_z; idx++) {
        int i = idx / (grid.dim_y * grid.dim_z); // i index
        int j = (idx / grid.dim_z) % grid.dim_y; // j index
        int k = idx % grid.dim_z;               // k index

        GridNode& node = grid.nodes[i][j][k];

        if (fabs(node.m) > std::numeric_limits<float>::epsilon()) {
            node.dLdp = node.dLdv / node.m;
            node.dLdm = -1.f / node.m * node.v.dot(node.dLdv);
        }
    }

    // Back prop P2G
#pragma omp parallel for //schedule(guided, 8)
    for (int p = 0; p < sz; p++) {
        const MaterialPoint& mp = pc.points[p];
        MaterialPoint& mp_prev = pc_prev.points[p];

        const Vec3& xp = mp_prev.x;
        const Mat3 F_transpose = (mp_prev.F + mp_prev.dFc).transpose();

        auto nodes = grid.QueryPoint_CubicBSpline(xp);

#pragma omp parallel for
        for (int i = 0; i < nodes.size(); i++) {
            GridNode& node = nodes[i];

            const Vec3& xg = node.x;
            Vec3 dgp = xg - xp;

            // Precompute the bspline and bspline slope values
            float dgp0_dx = dgp[0] / dx;
            float dgp1_dx = dgp[1] / dx;
            float dgp2_dx = dgp[2] / dx;

            float bspline_dgp0 = CubicBSpline(dgp0_dx);
            float bspline_dgp1 = CubicBSpline(dgp1_dx);
            float bspline_dgp2 = CubicBSpline(dgp2_dx);

            float bspline_slope_dgp0 = CubicBSplineSlope(dgp0_dx);
            float bspline_slope_dgp1 = CubicBSplineSlope(dgp1_dx);
            float bspline_slope_dgp2 = CubicBSplineSlope(dgp2_dx);

            float wgp = bspline_dgp0 * bspline_dgp1 * bspline_dgp2;

            Vec3 wgpGrad = -1.f / dx * Vec3(
                bspline_slope_dgp0 * bspline_dgp1 * bspline_dgp2,
                bspline_dgp0 * bspline_slope_dgp1 * bspline_dgp2,
                bspline_dgp0 * bspline_dgp1 * bspline_slope_dgp2
            );

            Mat3 G = -3.f / (dx * dx) * dt * mp_prev.vol * mp_prev.P * F_transpose + mp_prev.m * mp_prev.C;

            mp_prev.dLdP -= wgp * 3.f / (dx * dx) * dt * mp_prev.vol * node.dLdp * (F_transpose * dgp).transpose();

            mp_prev.dLdF -= wgp * 3.f / (dx * dx) * dt * mp_prev.vol * dgp * (mp_prev.P.transpose() * node.dLdp).transpose();

            mp_prev.dLdC += wgp * mp_prev.m * node.dLdp * dgp.transpose();

            mp_prev.dLdx += mp_prev.m * node.dLdm * wgpGrad;
            mp_prev.dLdx += (wgpGrad * (mp_prev.m * mp_prev.v + G * dgp).transpose() - wgp * G.transpose()) * node.dLdp;
            mp_prev.dLdx += wgpGrad * node.v.transpose() * mp_prev.dLdv_next;

            Vec3 temp = InnerProduct(mp_prev.dLdC_next, node.v * dgp.transpose()) * wgpGrad;
            temp -= wgp * mp_prev.dLdC_next.transpose() * node.v;

            mp_prev.dLdx += 3.f / (dx * dx) * temp;

            mp_prev.dLdv += wgp * mp_prev.m * (1.f - dt * drag) * node.dLdp;
        }

        // Back prop P_op_1
        mp_prev.dLdF += (Mat3::Identity() + dt * pc.points[p].C).transpose() * pc.points[p].dLdF;
        mp_prev.dLdF += d2_FCE_psi_dF2_mult_by_dF(mp_prev.F + mp_prev.dFc, mp_prev.lam, mp_prev.mu, mp_prev.dLdP);
        mp_prev.dLdx += pc.points[p].dLdx;
    }
}


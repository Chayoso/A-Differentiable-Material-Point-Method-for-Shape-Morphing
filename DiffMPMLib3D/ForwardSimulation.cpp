#include "pch.h"
#include "ForwardSimulation.h"
#include "Elasticity.h"
#include "Interpolation.h"

void DiffMPMLib3D::SingleThreadMPM::SingleParticle_op_1(MaterialPoint& mp)
{
    mp.P = PK_FixedCorotatedElasticity(mp.F + mp.dFc, mp.lam, mp.mu);
}

void DiffMPMLib3D::SingleThreadMPM::SingleParticle_to_grid(const MaterialPoint& mp, Grid& grid, float dt, float drag)
{
    float dx = grid.dx;

    Vec3 relative_point = mp.x - grid.min_point;
    int bot_left_index[3];

    for (int i = 0; i < 3; i++) {
        bot_left_index[i] = (int)std::floor(relative_point[i] / grid.dx) - 1;
    }

#pragma omp parallel for // schedule(guided, 24) 
    for (int idx = 0; idx <= 64; idx++) { // 0 to 4^3, since i, j, k go from 0 to 3
        int i = idx / 16;  // Determines the layer
        int j = (idx % 16) / 4;  // Determines the row within the layer
        int k = idx % 4;  // Determines the column within the row

        int index[3] = {
            bot_left_index[0] + i,
            bot_left_index[1] + j,
            bot_left_index[2] + k
        };

        if (0 <= index[0] && index[0] < grid.dim_x &&
            0 <= index[1] && index[1] < grid.dim_y &&
            0 <= index[2] && index[2] < grid.dim_z)
        {
            GridNode& node = grid.nodes[index[0]][index[1]][index[2]];
            Vec3 xg = node.x;
            Vec3 xp = mp.x;
            Vec3 dgp = xg - xp;

            float wgp = CubicBSpline(dgp[0] / grid.dx) * CubicBSpline(dgp[1] / grid.dx) * CubicBSpline(dgp[2] / grid.dx);

#pragma omp atomic
            node.m += wgp * mp.m;

            node.p += wgp * (mp.m * mp.v * (1.f - dt * drag) + (-3.f / (dx * dx) * dt * mp.vol * mp.P * (mp.F + mp.dFc).transpose() + mp.m * mp.C) * dgp);
        }
    }
}

void DiffMPMLib3D::SingleThreadMPM::SingleNode_op(GridNode& node, float dt, Vec3 f_ext)
{
    if (fabs(node.m) > std::numeric_limits<float>::epsilon()) {
        node.v = node.p / node.m + dt * f_ext;
    }
}

void DiffMPMLib3D::SingleThreadMPM::Grid_to_SingleParticle(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, const Grid& grid)
{
    float dx = grid.dx;
    next_timestep_mp.v.setZero();
    next_timestep_mp.C.setZero();

    Vec3 relative_point = curr_timestep_mp.x - grid.min_point;
    int bot_left_index[3];

    for (int i = 0; i < 3; i++) {
        bot_left_index[i] = (int)std::floor(relative_point[i] / grid.dx) - 1;
    }

    Vec3 local_v = Vec3(0.f, 0.f, 0.f);
    Mat3 local_C = Mat3::Zero();

#pragma omp parallel for
    for (int idx = 0; idx < 64; idx++) { // i, j, k ranges from 0 to 3, total 4*4*4 iterations
        int i = idx / 16; // Layer index
        int j = (idx % 16) / 4; // Row index within the layer
        int k = idx % 4; // Column index within the row

        int index[3] = {
            bot_left_index[0] + i,
            bot_left_index[1] + j,
            bot_left_index[2] + k
        };

        if (0 <= index[0] && index[0] < grid.dim_x &&
            0 <= index[1] && index[1] < grid.dim_y &&
            0 <= index[2] && index[2] < grid.dim_z)
        {
            const GridNode& node = grid.nodes[index[0]][index[1]][index[2]];
            Vec3 xg = node.x;
            Vec3 xp = curr_timestep_mp.x;
            Vec3 dgp = xg - xp;
            float wgp = CubicBSpline(dgp[0] / grid.dx) * CubicBSpline(dgp[1] / grid.dx) * CubicBSpline(dgp[2] / grid.dx);

            // Accumulate local values
            local_v += wgp * node.v;
            local_C += 3.0f / (grid.dx * grid.dx) * wgp * node.v * dgp.transpose();
        }
    }

    next_timestep_mp.v += local_v;
    next_timestep_mp.C += local_C;
}

void DiffMPMLib3D::SingleThreadMPM::SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float dt)
{
    next_timestep_mp.F = (Mat3::Identity() + dt * next_timestep_mp.C) * (curr_timestep_mp.F + curr_timestep_mp.dFc);
    next_timestep_mp.x = curr_timestep_mp.x + dt * next_timestep_mp.v;
}

void DiffMPMLib3D::SingleThreadMPM::smooth_deformation_gradient(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor) {
    next_timestep_mp.F = next_timestep_mp.F * (1.0f - smoothing_factor) + curr_timestep_mp.F * smoothing_factor;
}

void DiffMPMLib3D::SingleThreadMPM::SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor, float dt) {
    next_timestep_mp.F = (Mat3::Identity() + dt * next_timestep_mp.C) * (curr_timestep_mp.F + curr_timestep_mp.dFc);

    smooth_deformation_gradient(next_timestep_mp, curr_timestep_mp, smoothing_factor);

    next_timestep_mp.x = curr_timestep_mp.x + dt * next_timestep_mp.v;
}

void DiffMPMLib3D::SingleThreadMPM::ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float dt, float drag, Vec3 f_ext)
{
    P_op_1(curr_point_cloud);
    G_Reset(grid);
    P2G(curr_point_cloud, grid, dt, drag);
    G_op(grid, dt, f_ext);
    G2P(next_point_cloud, curr_point_cloud, grid);
    P_op_2(next_point_cloud, curr_point_cloud, dt);
}

void DiffMPMLib3D::SingleThreadMPM::ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float smoothing_factor, float dt, float drag, Vec3 f_ext)
{
    P_op_1(curr_point_cloud);
    G_Reset(grid);
    P2G(curr_point_cloud, grid, dt, drag);

//#pragma omp parallel for
//    for (int i = 0; i < grid.dim_x; i++) {
//        for (int j = 0; j < grid.dim_y; j++) {
//            for (int k = 0; k < grid.dim_z; k++) {
//                float& mref = grid.nodes[i][j][k].m;
//                if (mref < 0.f) {
//                    mref = 0.f;  // ¶Ç´Â std::max(0.f, mref);
//                }
//            }
//        }
//    }

    G_op(grid, dt, f_ext);
    G2P(next_point_cloud, curr_point_cloud, grid);
    P_op_2(next_point_cloud, curr_point_cloud, smoothing_factor, dt);
}

void DiffMPMLib3D::SingleThreadMPM::P_op_1(PointCloud& curr_point_cloud)
{
#pragma omp parallel for // schedule(guided, 24) 
    for (int p = 0; p < curr_point_cloud.points.size(); p++) {
        MaterialPoint& mp = curr_point_cloud.points[p];

        SingleParticle_op_1(mp);
    }
}

void DiffMPMLib3D::SingleThreadMPM::G_Reset(Grid& grid)
{
    grid.ResetValues();
}

void DiffMPMLib3D::SingleThreadMPM::P2G(const PointCloud& curr_point_cloud, Grid& grid, float dt, float drag)
{
#pragma omp parallel for // schedule(guided, 24) 
    for (int p = 0; p < curr_point_cloud.points.size(); p++) {
        const MaterialPoint& mp = curr_point_cloud.points[p];

        SingleParticle_to_grid(mp, grid, dt, drag);
    }
}

void DiffMPMLib3D::SingleThreadMPM::G_op(Grid& grid, float dt, Vec3 f_ext)
{
    int dim_x = grid.dim_x;
    int dim_y = grid.dim_y;
    int dim_z = grid.dim_z;
    int total_size = dim_x * dim_y * dim_z;

#pragma omp parallel for // schedule(guided, 24) 
    for (int idx = 0; idx < total_size; idx++) {
        int i = idx / (dim_y * dim_z);
        int j = (idx / dim_z) % dim_y;
        int k = idx % dim_z;

        GridNode& node = grid.nodes[i][j][k];
        SingleNode_op(node, dt, f_ext);
    }
}

void DiffMPMLib3D::SingleThreadMPM::G_op(Grid& grid, float dt, Vec3 gravity_point, float gravity_mag)
{
    int total_size = grid.dim_x * grid.dim_y * grid.dim_z;

#pragma omp parallel for // schedule(guided, 24) 
    for (int idx = 0; idx < total_size; idx++) {
        int i = idx / (grid.dim_y * grid.dim_z);          // i index
        int j = (idx / grid.dim_z) % grid.dim_y;          // j index
        int k = idx % grid.dim_z;                         // k index

        GridNode& node = grid.nodes[i][j][k];

        Vec3 f_ext = gravity_point - node.x;
        if (f_ext.isZero()) {
            f_ext.setZero();
        }
        else {
            f_ext.normalize();
            f_ext *= gravity_mag;
        }
        SingleNode_op(node, dt, f_ext);
    }
}

void DiffMPMLib3D::SingleThreadMPM::G2P(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, Grid& grid)
{
    float dx = grid.dx;

#pragma omp parallel for // schedule(guided, 24) 
    for (int p = 0; p < curr_point_cloud.points.size(); p++) {
        const MaterialPoint& curr_mp = curr_point_cloud.points[p];
        MaterialPoint& next_mp = next_point_cloud.points[p];

        Grid_to_SingleParticle(next_mp, curr_mp, grid);
    }
}

void DiffMPMLib3D::SingleThreadMPM::P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float dt)
{
#pragma omp parallel for // schedule(guided, 24) 
    for (int p = 0; p < next_point_cloud.points.size(); p++) {
        const MaterialPoint& curr_mp = curr_point_cloud.points[p];
        MaterialPoint& next_mp = next_point_cloud.points[p];

        SingleParticle_op_2(next_mp, curr_mp, dt);
    }
}

void DiffMPMLib3D::SingleThreadMPM::P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float smoothing_factor, float dt)
{
#pragma omp parallel for // schedule(guided, 24) 
    for (int p = 0; p < next_point_cloud.points.size(); p++) {
        const MaterialPoint& curr_mp = curr_point_cloud.points[p];
        MaterialPoint& next_mp = next_point_cloud.points[p];

        SingleParticle_op_2(next_mp, curr_mp, smoothing_factor, dt);
    }
}

/* 06/24 -- reduction */
void DiffMPMLib3D::SingleThreadMPM::CalculatePointCloudVolumes(PointCloud& curr_point_cloud, Grid& grid)
{
    float dx = grid.dx;
    P2G(curr_point_cloud, grid, 0.f, 0.f);

#pragma omp parallel for //schedule(guided, 24) 
    for (int p = 0; p < curr_point_cloud.points.size(); p++) {
        MaterialPoint& mp = curr_point_cloud.points[p];

        std::vector<std::array<int, 3>> indices;
        auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);

        float mass_from_grid = 0.f;

        for (int i = 0; i < nodes.size(); i++) {
            GridNode& node = nodes[i];

            Vec3 xg = node.x;
            Vec3 xp = mp.x;
            Vec3 dgp = xg - xp;
            float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

#pragma omp atomic
            mass_from_grid += wgp * node.m;
        }

        mp.vol = (fabs(mass_from_grid) > std::numeric_limits<float>::epsilon()) ? (mp.m / mass_from_grid) : 0.f;
    }
}

void DiffMPMLib3D::SingleThreadMPM::P2G_Mass(const std::vector<Vec3> points, Grid& grid, float mp_m)
{
    float dx = grid.dx;

    int total_size = grid.dim_x * grid.dim_y * grid.dim_z;

#pragma omp parallel for
    for (int idx = 0; idx < total_size; idx++) {
        int i = idx / (grid.dim_y * grid.dim_z);          // i index
        int j = (idx / grid.dim_z) % grid.dim_y;          // j index
        int k = idx % grid.dim_z;                         // k index

        grid.nodes[i][j][k].m = 0.f;
    }

#pragma omp parallel for collapse(2)
    for (int p = 0; p < points.size(); p++) {
        const Vec3& mp_x = points[p];

        std::vector<std::array<int, 3>> indices;
        auto nodes = grid.QueryPoint_CubicBSpline(mp_x, &indices);

        for (int i = 0; i < nodes.size(); i++) {
            GridNode& node = nodes[i];

            Vec3 xg = node.x;
            Vec3 xp = mp_x;
            Vec3 dgp = xg - xp;
            float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

            // MLS-MPM APIC
#pragma omp atomic
            node.m += wgp * mp_m;
        }
    }
}

void DiffMPMLib3D::SingleThreadMPM::G2G_Mass(Grid& grid_1, Grid& grid_2)
{
    float dx = grid_1.dx;

#pragma omp parallel for collapse(3)
    for (int i = 0; i < grid_2.dim_x; i++) {
        for (int j = 0; j < grid_2.dim_y; j++) {
            for (int k = 0; k < grid_2.dim_z; k++) {
                grid_2.nodes[i][j][k].m = 0.f;
            }
        }
    }

#pragma omp parallel for collapse(4)
    for (int i = 0; i < grid_2.dim_x; i++) {
        for (int j = 0; j < grid_2.dim_y; j++) {
            for (int k = 0; k < grid_2.dim_z; k++) {
                float& mp_m = grid_2.nodes[i][j][k].m;
                mp_m = 0.f;
                const Vec3& mp_x = grid_2.nodes[i][j][k].x;

                std::vector<std::array<int, 3>> indices;
                auto nodes = grid_1.QueryPoint_CubicBSpline(mp_x, &indices);
                for (size_t i = 0; i < nodes.size(); i++) {
                    const GridNode& node = nodes[i];

                    Vec3 xg = node.x;
                    Vec3 xp = mp_x;
                    Vec3 dgp = xg - xp;
                    float wgp = CubicBSpline(dgp[0] / dx) * CubicBSpline(dgp[1] / dx) * CubicBSpline(dgp[2] / dx);

                    // MLS-MPM APIC
#pragma omp atomic
                    mp_m += wgp * node.m;
                }
            }
        }
    }
}

#pragma once

#include "PointCloud.h"
#include "Grid.h"

#pragma once
namespace DiffMPMLib3D {
	namespace SingleThreadMPM {

		void SingleParticle_op_1(MaterialPoint& mp);
		void SingleParticle_to_grid(const MaterialPoint& mp, Grid& grid, float dt, float drag);
		void SingleNode_op(GridNode& node, float dt, Vec3 f_ext);
		void Grid_to_SingleParticle(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, const Grid& grid);
		void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float dt);
		void SingleParticle_op_2(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor, float dt);

		void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float dt, float drag, Vec3 f_ext);
		void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& grid, float smoothing_factor, float dt, float drag, Vec3 f_ext);

		void smooth_deformation_gradient(MaterialPoint& next_timestep_mp, const MaterialPoint& curr_timestep_mp, float smoothing_factor);


		void P_op_1(PointCloud& curr_point_cloud);

		void G_Reset(Grid& grid);

		void P2G(const PointCloud& curr_point_cloud, Grid& grid, float dt, float drag);

		void G_op(Grid& grid, float dt, Vec3 f_ext);
		void G_op(Grid& grid, float dt, Vec3 gravity_point, float gravity_mag);

		void G2P(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, Grid& grid);

		void P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float dt);
		void P_op_2(PointCloud& next_point_cloud, const PointCloud& curr_point_cloud, float smoothing_factor, float dt);


		// UTILITY
		void CalculatePointCloudVolumes(PointCloud& curr_point_cloud, Grid& grid);

		void P2G_Mass(const std::vector<Vec3> points, Grid& grid, float mp_m);

		void G2G_Mass(Grid& grid_1, Grid& grid_2);

	}
}
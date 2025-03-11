#pragma once
#include "PointCloud.h"
#include "Grid.h"

namespace DiffMPMLib3D {
	namespace MultiThreadMPM {
		void ForwardTimeStep(PointCloud& next_point_cloud, PointCloud& curr_point_cloud, Grid& main_grid,
			std::vector<std::shared_ptr<Grid>>& proxy_grids, float dt, float drag, Vec3 f_ext);

		// parallelized over sections of points
		void Multi_P_op_1(std::vector<MaterialPoint>& points, size_t p_ind_start, size_t num_p);

		// parallelized over sections of points
		void Multi_P2G(const std::vector<MaterialPoint>& points, Grid& proxy_grid, size_t p_ind_start, size_t num_p, float dt, float drag);

		// parallelized over sections of grid nodes
		// does reduction and grid update
		void Multi_G_op(const std::vector<std::shared_ptr<Grid>>& proxy_grids, Grid& main_grid,
			size_t block_start_i, size_t block_start_j, size_t block_start_k,
			size_t block_size_i, size_t block_size_j, size_t block_size_k,
			float dt, Vec3 f_ext);

		// parallelized over sections of points
		void Multi_G2P(std::vector<MaterialPoint>& next_points, const std::vector<MaterialPoint>& curr_points, const Grid& main_grid,
			size_t p_ind_start, size_t num_p);

		// parallelized over sections of points
		void Multi_P_op_2(std::vector<MaterialPoint>& next_points, const std::vector<MaterialPoint>& curr_points, size_t p_ind_start, size_t num_p, float dt);
	}
}
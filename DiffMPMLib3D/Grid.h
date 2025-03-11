#pragma once
#include "GridNode.h"
#include <functional>
#include "Backprop.cuh"

namespace DiffMPMLib3D {
	struct Grid
	{
		/*
		* Grid stores a 3D uniform grid of nodal interpolation points.
		* The convention is that i, j, k corresponds the the positive x, y, z axes.
		*/

		Grid(int _dim_x, int _dim_y, int dim_z, float _dx, Vec3 _min_point);
		Grid(const Grid& grid);

		//std::vector<std::reference_wrapper<GridNode>> QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr);
		std::vector<std::reference_wrapper<DiffMPMLib3D::GridNode>> DiffMPMLib3D::Grid::QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr);
		std::vector<std::reference_wrapper<const GridNode>> QueryPointConst_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices = nullptr) const;
		//std::vector<DiffMPMLib3D::GridNode*> Grid::QueryPoint_CubicBSpline_with_kernel(Vec3 point, std::vector<std::array<int, 3>>* indices);

		std::vector<GridNode> DiffMPMLib3D::Grid::flatten() const {
			std::vector<GridNode> flat_nodes(dim_x * dim_y * dim_z);
			for (int idx = 0; idx < dim_x * dim_y * dim_z; ++idx) {
				int x = idx / (dim_y * dim_z);
				int y = (idx / dim_z) % dim_y;
				int z = idx % dim_z;
				flat_nodes[idx] = nodes[x][y][z];
			}
			return flat_nodes;
		}

		void DiffMPMLib3D::Grid::unflatten(const std::vector<GridNode>& flat_nodes) {
			nodes = std::vector<std::vector<std::vector<GridNode>>>(dim_x, std::vector<std::vector<GridNode>>(dim_y, std::vector<GridNode>(dim_z)));
			for (int idx = 0; idx < flat_nodes.size(); ++idx) {
				int x = idx / (dim_y * dim_z);
				int y = (idx / dim_z) % dim_y;
				int z = idx % dim_z;
				nodes[x][y][z] = flat_nodes[idx];
			}
		}

		std::vector<Vec3> GetNodePositions() const;
		std::vector<float> GetNodeMasses() const;
		std::vector<Vec3> GetNodeVelocities() const;
		void GetMassSDF(Eigen::MatrixXf& GV, Eigen::VectorXf& Gf) const;

		void ResetGradients();
		void ResetValues();

		int dim_x;
		int dim_y;
		int dim_z;
		float dx;
		Vec3 min_point;
		std::vector<std::vector<std::vector<GridNode>>> nodes;
	};
}
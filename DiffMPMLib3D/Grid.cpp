#include "Grid.h"

DiffMPMLib3D::Grid::Grid(int _dim_x, int _dim_y, int _dim_z, float _dx, Vec3 _min_point)
	: dim_x(_dim_x), dim_y(_dim_y), dim_z(_dim_z), dx(_dx), min_point(_min_point)
{
	nodes = std::vector<std::vector<std::vector<GridNode>>>();
	nodes.resize(dim_x);

#pragma omp parallel for
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx / dim_z) % dim_y;
		int k = idx % dim_z;

		if (nodes[i].size() < dim_y) {
			nodes[i].resize(dim_y);
		}
		if (nodes[i][j].size() < dim_z) {
			nodes[i][j].resize(dim_z);
		}

		nodes[i][j][k].x = min_point + Vec3(i, j, k) * dx;
	}

	auto total_bytes = sizeof(GridNode) * dim_x * dim_y * dim_z;
	total_bytes = total_bytes >> 10;
	std::cout << "generated grid of size: " << total_bytes << " KB" << std::endl;
}

DiffMPMLib3D::Grid::Grid(const Grid& grid)
{
	dim_x = grid.dim_x;
	dim_y = grid.dim_y;
	dim_z = grid.dim_z;
	dx = grid.dx;
	min_point = grid.min_point;
	nodes = grid.nodes;
}

std::vector<std::reference_wrapper<DiffMPMLib3D::GridNode>> DiffMPMLib3D::Grid::QueryPoint_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices)
{
	/*
	* Returns all the nodes which are within interpolation range of the point position
	*/

	std::vector<std::reference_wrapper<GridNode>> ret;

	Vec3 relative_point = point - min_point;

	int bot_left_index[3];
	for (size_t i = 0; i < 3; i++) {
		bot_left_index[i] = (int)std::floor(relative_point[i] / dx) - 1;
	}

#pragma omp parallel for collapse(3)
	for (int i = 0; i <= 3; i++) {
		for (int j = 0; j <= 3; j++) {
			for (int k = 0; k <= 3; k++) {
				int index[3] = {
				   bot_left_index[0] + i,
				   bot_left_index[1] + j,
				   bot_left_index[2] + k
				};

				// check if this is a valid node
				if (0 <= index[0] && index[0] < dim_x &&
					0 <= index[1] && index[1] < dim_y &&
					0 <= index[2] && index[2] < dim_z)
				{
					ret.push_back(std::ref(nodes[index[0]][index[1]][index[2]]));

					if (indices != nullptr) {
						indices->push_back({ index[0], index[1], index[2] });
					}
				}
			}
		}
	}

	return ret;
}

//std::vector<DiffMPMLib3D::GridNode*> DiffMPMLib3D::Grid::QueryPoint_CubicBSpline_with_kernel(Vec3 point, std::vector<std::array<int, 3>>* indices) {
//	int range = 4;
//	int total_elements = range * range * range;
//
//	GridNode* d_nodes;
//	GridNode** d_result_nodes;
//	int* d_result_indices = nullptr;
//
//	cudaError_t err;
//
//	err = cudaMalloc(&d_nodes, dim_x * dim_y * dim_z * sizeof(GridNode));
//	if (err != cudaSuccess) {
//		std::cerr << "CUDA malloc failed for d_nodes: " << cudaGetErrorString(err) << std::endl;
//		return {};
//	}
//
//	err = cudaMalloc(&d_result_nodes, total_elements * sizeof(GridNode*));
//	if (err != cudaSuccess) {
//		std::cerr << "CUDA malloc failed for d_result_nodes: " << cudaGetErrorString(err) << std::endl;
//		cudaFree(d_nodes);
//		return {};
//	}
//
//	if (indices) {
//		err = cudaMalloc(&d_result_indices, total_elements * 3 * sizeof(int));
//		if (err != cudaSuccess) {
//			std::cerr << "CUDA malloc failed for d_result_indices: " << cudaGetErrorString(err) << std::endl;
//			cudaFree(d_nodes);
//			cudaFree(d_result_nodes);
//			return {};
//		}
//	}
//
//	// 데이터 복사
//	std::vector<GridNode> flat_nodes(dim_x * dim_y * dim_z);
//	for (int x = 0; x < dim_x; ++x) {
//		for (int y = 0; y < dim_y; ++y) {
//			for (int z = 0; z < dim_z; ++z) {
//				flat_nodes[x + y * dim_x + z * dim_x * dim_y] = nodes[x][y][z];
//			}
//		}
//	}
//
//	err = cudaMemcpy(d_nodes, flat_nodes.data(), flat_nodes.size() * sizeof(GridNode), cudaMemcpyHostToDevice);
//	if (err != cudaSuccess) {
//		std::cerr << "CUDA memcpy failed for d_nodes: " << cudaGetErrorString(err) << std::endl;
//		cudaFree(d_nodes);
//		cudaFree(d_result_nodes);
//		if (indices) {
//			cudaFree(d_result_indices);
//		}
//		return {};
//	}
//
//	// CUDA 커널 호출
//	int threadsPerBlock = 256;
//	int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
//
//	QueryPoint_CubicBSpline_kernel << <blocksPerGrid, threadsPerBlock >> > (
//		point, min_point, dx, dim_x, dim_y, dim_z, d_nodes, d_result_nodes, d_result_indices, range);
//
//	err = cudaDeviceSynchronize();
//	if (err != cudaSuccess) {
//		std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
//		cudaFree(d_nodes);
//		cudaFree(d_result_nodes);
//		if (indices) {
//			cudaFree(d_result_indices);
//		}
//		return {};
//	}
//
//	// 결과 복사
//	std::vector<GridNode*> result_nodes(total_elements);
//	err = cudaMemcpy(result_nodes.data(), d_result_nodes, total_elements * sizeof(GridNode*), cudaMemcpyDeviceToHost);
//	if (err != cudaSuccess) {
//		std::cerr << "CUDA memcpy failed for d_result_nodes: " << cudaGetErrorString(err) << std::endl;
//		cudaFree(d_nodes);
//		cudaFree(d_result_nodes);
//		if (indices) {
//			cudaFree(d_result_indices);
//		}
//		return {};
//	}
//
//	std::vector<GridNode> host_result_nodes(total_elements);
//	for (int i = 0; i < total_elements; ++i) {
//		if (result_nodes[i] != nullptr) {
//			err = cudaMemcpy(&host_result_nodes[i], result_nodes[i], sizeof(GridNode), cudaMemcpyDeviceToHost);
//			if (err != cudaSuccess) {
//				std::cerr << "CUDA memcpy failed for host_result_nodes: " << cudaGetErrorString(err) << std::endl;
//				cudaFree(d_nodes);
//				cudaFree(d_result_nodes);
//				if (indices) {
//					cudaFree(d_result_indices);
//				}
//				return {};
//			}
//		}
//	}
//
//	std::vector<GridNode*> valid_nodes;
//
//	if (indices) {
//		std::vector<int> result_indices(total_elements * 3);
//		err = cudaMemcpy(result_indices.data(), d_result_indices, total_elements * 3 * sizeof(int), cudaMemcpyDeviceToHost);
//		if (err != cudaSuccess) {
//			std::cerr << "CUDA memcpy failed for d_result_indices: " << cudaGetErrorString(err) << std::endl;
//			cudaFree(d_nodes);
//			cudaFree(d_result_nodes);
//			cudaFree(d_result_indices);
//			return {};
//		}
//
//		indices->clear();
//		for (int i = 0; i < total_elements; ++i) {
//			if (result_indices[i * 3] != -1 && result_nodes[i] != nullptr) {
//				indices->emplace_back(std::array<int, 3>{ result_indices[i * 3], result_indices[i * 3 + 1], result_indices[i * 3 + 2] });
//				valid_nodes.emplace_back(&host_result_nodes[i]);
//			}
//		}
//	}
//	else {
//		for (int i = 0; i < total_elements; ++i) {
//			if (result_nodes[i] != nullptr) {
//				valid_nodes.emplace_back(&host_result_nodes[i]);
//			}
//		}
//	}
//
//	// 메모리 해제
//	cudaFree(d_nodes);
//	cudaFree(d_result_nodes);
//	if (d_result_indices) {
//		cudaFree(d_result_indices);
//	}
//
//	return valid_nodes;
//}


std::vector<std::reference_wrapper<const DiffMPMLib3D::GridNode>> DiffMPMLib3D::Grid::QueryPointConst_CubicBSpline(Vec3 point, std::vector<std::array<int, 3>>* indices) const
{
	/*
	* Returns all the nodes which are within interpolation range of the point position
	*/

	std::vector<std::reference_wrapper<const GridNode>> ret;

	Vec3 relative_point = point - min_point;

	int bot_left_index[3];
	for (size_t i = 0; i < 3; i++) {
		bot_left_index[i] = (int)std::floor(relative_point[i] / dx) - 1;
	}

#pragma omp parallel for collapse(3)
	for (int i = 0; i <= 3; i++) {
		for (int j = 0; j <= 3; j++) {
			for (int k = 0; k <= 3; k++) {
				int index[3] = {
					bot_left_index[0] + i,
					bot_left_index[1] + j,
					bot_left_index[2] + k
				};

				// check if this is a valid node
				if (0 <= index[0] && index[0] < dim_x &&
					0 <= index[1] && index[1] < dim_y &&
					0 <= index[2] && index[2] < dim_z)
				{
					ret.push_back(std::cref(nodes[index[0]][index[1]][index[2]]));

					if (indices != nullptr) {
						indices->push_back({ index[0], index[1], index[2] });
					}
				}
			}
		}
	}

	return ret;
}



std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::Grid::GetNodePositions() const
{
	std::vector<decltype(nodes[0][0][0].x)> ret(dim_x * dim_y * dim_z);

#pragma omp parallel for
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx / dim_z) % dim_y;
		int k = idx % dim_z;

		ret[idx] = nodes[i][j][k].x;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::Grid::GetNodeMasses() const
{
	std::vector<decltype(nodes[0][0][0].m)> ret(dim_x * dim_y * dim_z);

#pragma omp parallel for
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx / dim_z) % dim_y;
		int k = idx % dim_z;

		ret[idx] = nodes[i][j][k].m;
	}

	return ret;
}

std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::Grid::GetNodeVelocities() const
{
	std::vector<decltype(nodes[0][0][0].v)> ret(dim_x * dim_y * dim_z);

#pragma omp parallel for
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx / dim_z) % dim_y;
		int k = idx % dim_z;

		ret[idx] = nodes[i][j][k].v;
	}

	return ret;
}

void DiffMPMLib3D::Grid::GetMassSDF(Eigen::MatrixXf& GV, Eigen::VectorXf& Gf) const
{
	GV.resize(dim_x * dim_y * dim_z, 3);
	Gf.resize(dim_x * dim_y * dim_z);

	int index = 0;

#pragma omp parallel for collapse(3)
	for (int i = 0; i < dim_x; i++) {
		for (int j = 0; j < dim_y; j++) {
			for (int k = 0; k < dim_z; k++) {
				GV.row(index) = nodes[i][j][k].x;
				Gf(index) = nodes[i][j][k].m;
				index++;
			}
		}
	}
	return;
}

void DiffMPMLib3D::Grid::ResetGradients()
{
#pragma omp parallel for
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx % (dim_y * dim_z)) / dim_z;
		int k = idx % dim_z;

		nodes[i][j][k].ResetGradients();
	}
}

void DiffMPMLib3D::Grid::ResetValues()
{	
#pragma omp parallel for 
	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
		int i = idx / (dim_y * dim_z);
		int j = (idx % (dim_y * dim_z)) / dim_z;
		int k = idx % dim_z;

		nodes[i][j][k].m = 0.0;
		nodes[i][j][k].v.setZero();
		nodes[i][j][k].p.setZero();
	}

}
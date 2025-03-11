#pragma once
#include "pch.h"

namespace DiffMPMLib3D {
	namespace GeometryLoading
	{

		std::vector<Vec3> GeneratePointCloudFromWatertightTriangleMesh(
			const Eigen::MatrixXf& V,
			const Eigen::MatrixXi& F,
			Vec3 min_point,
			Vec3 max_point,
			float sampling_dx
		);
	}
}
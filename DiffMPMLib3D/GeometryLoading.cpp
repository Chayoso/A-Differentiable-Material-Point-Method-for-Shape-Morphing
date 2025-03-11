#include "pch.h"
#include "GeometryLoading.h"
#include "igl/point_mesh_squared_distance.h"
#include "igl/signed_distance.h"
#include <vector>
#include <Eigen/Core>
#include <cmath>
#include <iostream>

std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::GeometryLoading::GeneratePointCloudFromWatertightTriangleMesh(
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXi& F,
    Vec3 min_point,
    Vec3 max_point,
    float sampling_dx)
{
    using namespace Eigen;

    // 1. Generate uniform grid sample points
    int dims[3];
    for (int i = 0; i < 3; i++) {
        dims[i] = static_cast<int>(std::ceil((max_point[i] - min_point[i]) / sampling_dx)) + 1; // +1 to include max_point
    }

    std::vector<Vec3> sample_points;
    sample_points.reserve(dims[0] * dims[1] * dims[2]);
#pragma parallel omp for collapse(3)
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                Vec3 point = min_point + sampling_dx * Vec3(float(i), float(j), float(k));
                sample_points.emplace_back(point);
            }
        }
    }

    // Convert sample_points to Eigen matrix
    MatrixXf P(sample_points.size(), 3);
    for (size_t i = 0; i < sample_points.size(); ++i) {
        P.row(i) = sample_points[i];
    }

    // Debug: Output the range of sample points
    std::cout << "Sample points range: ["
        << P.col(0).minCoeff() << ", " << P.col(0).maxCoeff() << "] x ["
        << P.col(1).minCoeff() << ", " << P.col(1).maxCoeff() << "] x ["
        << P.col(2).minCoeff() << ", " << P.col(2).maxCoeff() << "]" << std::endl;

    // Debug: Output the range of mesh vertices
    std::cout << "Mesh vertices range: ["
        << V.col(0).minCoeff() << ", " << V.col(0).maxCoeff() << "] x ["
        << V.col(1).minCoeff() << ", " << V.col(1).maxCoeff() << "] x ["
        << V.col(2).minCoeff() << ", " << V.col(2).maxCoeff() << "]" << std::endl;

    // 2. Get signed distances
    VectorXf S;
    VectorXi I;
    MatrixXf C;
    MatrixXf N;

    igl::SignedDistanceType sign_type = igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER;
    igl::signed_distance(P, V, F, sign_type, S, I, C, N);

    // Debug: Output some signed distances
    std::cout << "Signed distances: " << std::endl;
    for (int i = 0; i < std::min(10, (int)S.size()); ++i) {
        std::cout << "P[" << i << "] = " << P.row(i) << ", S = " << S[i] << std::endl;
    }

    // 3. Store all points with negative signed distance
    std::vector<Vec3> points;
    points.reserve(P.rows());
    for (int i = 0; i < P.rows(); i++) {
        if (S[i] <= 0.0) {
            points.push_back(P.row(i));
        }
    }

    // Debug: Output the number of points inside the mesh
    std::cout << "Number of points inside the mesh: " << points.size() << std::endl;

    return points;
}

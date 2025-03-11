#pragma once

#include "PointCloud.h"
#include "Grid.h"
#include "CompGraph.h"
#include "polyscope/point_cloud.h"
#include "cereal/archives/json.hpp"
#include "cereal_eigen.h"
using namespace DiffMPMLib3D;

const std::string PS_POINT_CLOUD_1 = "point cloud 1";
const std::string PS_POINT_CLOUD_2 = "point cloud 2";
const std::string PS_POINT_CLOUD_3 = "point cloud 3";
const std::string PS_SIM_GRID = "simulation grid";

struct OptInput
{
	std::string mpm_input_mesh_path = "experiments\\big_sca_demo\\input\\bob.obj";
	std::string mpm_target_mesh_path = "experiments\\big_sca_demo\\input\\spot.obj";

	// SIMULATION SETTINGS
	float grid_dx = 1.0;
	int points_per_cell_cuberoot = 2;

	/*Vec3 grid_min_point = Vec3(-8.0, -8.0, -8.0);
	Vec3 grid_max_point = Vec3(8.0, 8.0, 8.0);*/
	Vec3 grid_min_point = Vec3(-8.0, -8.0, -8.0) * 2.0;
	Vec3 grid_max_point = Vec3(8.0, 8.0, 8.0) * 2.0;

	// stiff material
	float lam = 58333.0f;
	float mu = 38888.9f;

	// fluid like material
	//float lam = 2500.0;
	//float mu = 100.0;
	float p_density = 40.0f;// *5.0;

	float dt = 1.f / 120.0f;
	float drag = 0.5f;
	Vec3 f_ext = Vec3(0.f, 0.f, 0.f);



	// OPTIMIZATION SETTINGS
	int num_animations = 40; // number of animations/episodes (which will be stitched together)
	int num_timesteps = 30; // number of timesteps per animation/episode
	int control_stride = 10; // period between control deformation gradient activations
	int max_gd_iters = 25; // gradient descent iterations
	int max_ls_iters = 15; // line search iterations
	float initial_alpha = 0.1f; // gradient descent step size
	float gd_tol = 1e-3f; // gradient descent tolerance for gradient norm
	float smoothing_factor = 0.1f;

	void ImGui();

	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(
			CEREAL_NVP(mpm_input_mesh_path),
			CEREAL_NVP(mpm_target_mesh_path),
			CEREAL_NVP(grid_dx),
			CEREAL_NVP(points_per_cell_cuberoot),
			CEREAL_NVP(grid_min_point),
			CEREAL_NVP(grid_max_point),
			CEREAL_NVP(lam),
			CEREAL_NVP(mu),
			CEREAL_NVP(p_density),
			CEREAL_NVP(dt),
			CEREAL_NVP(drag),
			CEREAL_NVP(f_ext),
			CEREAL_NVP(num_animations),
			CEREAL_NVP(num_timesteps),
			CEREAL_NVP(control_stride),
			CEREAL_NVP(max_gd_iters),
			CEREAL_NVP(max_ls_iters),
			CEREAL_NVP(initial_alpha),
			CEREAL_NVP(gd_tol),
			CEREAL_NVP(smoothing_factor)
		);
	}
};

bool LoadMPMPointCloudFromObj(
	std::string obj_path,
	std::shared_ptr<PointCloud>& mpm_point_cloud,
	float point_dx,
	float density,
	float lam,
	float mu
);

bool LoadScene(
	const OptInput& opt_input,
	std::shared_ptr<PointCloud>& mpm_point_cloud,
	std::shared_ptr<Grid>& mpm_grid,
	polyscope::PointCloud** polyscope_point_cloud,
	polyscope::PointCloud** polyscope_grid
);

bool LoadCompGraph(
	const OptInput& opt_input,
	std::shared_ptr<CompGraph>& comp_graph,
	polyscope::PointCloud** polyscope_input_point_cloud,
	polyscope::PointCloud** polyscope_target_point_cloud,
	polyscope::PointCloud** polyscope_grid
);
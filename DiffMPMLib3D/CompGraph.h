#pragma once
#include "pch.h"
#include "PointCloud.h"
#include "Grid.h"
#include <math.h>
#include <omp.h>
#include <iostream>
#include <vector>

namespace DiffMPMLib3D {
	struct CompGraphLayer
	{
		std::shared_ptr<PointCloud> point_cloud = nullptr;
		std::shared_ptr<Grid> grid = nullptr;
	};

	class CompGraph
	{
	public:
		CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid);


		void OptimizeDefGradControlSequence(
			// SIMULATION
			int num_steps, // number of timestepes, aka layers in the comp graph
			float _dt,
			float _drag,
			Vec3 _f_ext,

			// OPTIMIZATION
			int control_stride, // stride between control frames
			int max_gd_iters, // max gradient descent iterations per control frame
			int max_line_search_iters, // max line search iterations per control frame
			float initial_alpha, // initial gradient descent step size
			float gd_tol, // tolerance factor used for determining when gradient descent has converged
			float smoothing_factor
		);

			
		void SetUpCompGraph(int num_layers);

		float EndLayerMassLoss();

		void ComputeForwardPass(size_t start_layer);
		void ComputeBackwardPass(size_t control_layer);

		void FiniteDifferencesGradientTest(int num_steps, size_t particle_id);


		std::vector<CompGraphLayer> layers;

		std::shared_ptr<const Grid> target_grid;

	private:
		Vec3 f_ext = Vec3::Zero();
		float dt = 1.0f / 120.0f;
		float drag = 0.5f;
		float smoothing_factor = 0.1f;
		float previous_loss = std::numeric_limits<float>::max();

		float alpha_decay = 0.96; 
		float min_alpha = 1e-3; 
		std::vector<float> gradientNormHistory;

		int timestep = 1;

	};
}
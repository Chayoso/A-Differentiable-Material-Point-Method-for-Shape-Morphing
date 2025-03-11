#include "pch.h"
#include "CompGraph.h"
#include "ForwardSimulation.h"
#include "BackPropagation.h"
#include "Interpolation.h"

void DiffMPMLib3D::CompGraph::SetUpCompGraph(int num_layers)
{
	assert(num_layers > 0);
	layers.resize(num_layers);

#pragma omp parallel for //schedule(guided, 8)
	for (int i = 1; i < num_layers; i++) {
		layers[i].point_cloud = std::make_shared<PointCloud>(*layers.front().point_cloud);
		layers[i].grid = std::make_shared<Grid>(*layers.front().grid);
	}
}

// Evaluate loss third, and take average for current time step.
//float DiffMPMLib3D::CompGraph::EndLayerMassLoss()
//{
//	float out_of_target_penalty = 1.f;
//	float eps = 1e-4f;
//
//	PointCloud& point_cloud = *layers.back().point_cloud;
//	Grid&  grid = *layers.back().grid;
//	float dx = grid.dx;
//
//	assert(target_grid->dim_x == grid.dim_x &&
//		target_grid->dim_y == grid.dim_y &&
//		target_grid->dim_z == grid.dim_z
//	);
//
//	SingleThreadMPM::G_Reset(grid);
//	SingleThreadMPM::P2G(point_cloud, grid, 0.f, 0.f);
//
//	float loss = 0.f;
//
//	int dim_x = target_grid->dim_x;
//	int dim_y = target_grid->dim_y;
//	int dim_z = target_grid->dim_z;
//
//
//#pragma omp parallel for reduction(+:loss)
//	for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
//		int i = idx / (dim_y * dim_z);
//		int j = (idx / dim_z) % dim_y;
//		int k = idx % dim_z;
//
//		float dm = grid.nodes[i][j][k].m - target_grid->nodes[i][j][k].m;
//		float _clog = log(grid.nodes[i][j][k].m + 1.f + eps) - log(target_grid->nodes[i][j][k].m + 1.f + eps);
//		loss += (0.5f * _clog * _clog);
//
//		// dLdm
//		grid.nodes[i][j][k].dLdm = _clog / (grid.nodes[i][j][k].m + 1.f + eps);
//	}
//
//
//#pragma omp parallel for //schedule(guided, 8)
//	for (int p = 0; p < point_cloud.points.size(); p++) {
//		MaterialPoint& __restrict mp = point_cloud.points[p];
//		mp.dLdx.setZero();
//
//		std::vector<std::array<int, 3>> indices;
//		auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);
//
//		for (int i = 0; i < nodes.size(); i++) {
//			const GridNode& node = nodes[i];
//
//			Vec3 xg = node.x;
//			Vec3 xp = mp.x;
//			Vec3 dgp = xg - xp;
//
//			// Precompute the bspline and bspline slope values
//			float dgp0_dx = dgp[0] / dx;
//			float dgp1_dx = dgp[1] / dx;
//			float dgp2_dx = dgp[2] / dx;
//
//			float bspline_dgp0 = CubicBSpline(dgp0_dx);
//			float bspline_dgp1 = CubicBSpline(dgp1_dx);
//			float bspline_dgp2 = CubicBSpline(dgp2_dx);
//
//			float bspline_slope_dgp0 = CubicBSplineSlope(dgp0_dx);
//			float bspline_slope_dgp1 = CubicBSplineSlope(dgp1_dx);
//			float bspline_slope_dgp2 = CubicBSplineSlope(dgp2_dx);
//
//			float wgp = bspline_dgp0 * bspline_dgp1 * bspline_dgp2;
//
//			Vec3 wgpGrad = -1.f / dx * Vec3(
//				bspline_slope_dgp0 * bspline_dgp1 * bspline_dgp2,
//				bspline_dgp0 * bspline_slope_dgp1 * bspline_dgp2,
//				bspline_dgp0 * bspline_dgp1 * bspline_slope_dgp2
//			);
//
//			if (fabs(target_grid->nodes[indices[i][0]][indices[i][1]][indices[i][2]].m) > std::numeric_limits<float>::epsilon())
//				mp.dLdx += mp.m * node.dLdm * wgpGrad;
//			else
//				mp.dLdx += out_of_target_penalty * mp.m * node.dLdm * wgpGrad; // PENALTY FOR BEING OUTSIDE TARGET GRID
//		}
//
//		// other derivatives
//		mp.dLdF.setZero();
//		mp.dLdv.setZero();
//		mp.dLdC.setZero();
//	}
//
//	return loss;
//}

float DiffMPMLib3D::CompGraph::EndLayerMassLoss()
{
    float out_of_target_penalty = 1.f;

    // 로그 계산 시 사용될 eps
    float eps = 1e-4f;

    // -------------------------------
    // [소프트 최소질량(Soft Constraint) 설정]
    // -------------------------------
    // min_mass : 이 값 이하로 질량이 내려가면 큰 벌점을 부과
    // penalty_weight : 벌점 항의 세기를 조절
    float min_mass       = 1e-3f;
    float penalty_weight = 10.f;

    PointCloud& point_cloud = *layers.back().point_cloud;
    Grid&  grid = *layers.back().grid;
    float dx = grid.dx;

    assert(target_grid->dim_x == grid.dim_x &&
           target_grid->dim_y == grid.dim_y &&
           target_grid->dim_z == grid.dim_z);

    // 먼저 grid 초기화 & point_cloud -> grid로 질량(m) 전파
    SingleThreadMPM::G_Reset(grid);
    SingleThreadMPM::P2G(point_cloud, grid, 0.f, 0.f);

    float loss = 0.f;

    int dim_x = target_grid->dim_x;
    int dim_y = target_grid->dim_y;
    int dim_z = target_grid->dim_z;

#pragma omp parallel for reduction(+:loss)
    for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
        int i = idx / (dim_y * dim_z);
        int j = (idx / dim_z) % dim_y;
        int k = idx % dim_z;

        float c_m = grid.nodes[i][j][k].m;     // 현재 grid 상의 질량
        float t_m = target_grid->nodes[i][j][k].m; // 목표(grid) 질량

        // -------------------
        // (1) 로그 차 스쿼어(loss) 계산
        // -------------------
        //  log(current_m + 1 + eps) - log(target_m + 1 + eps)
        float _clog = std::log(c_m + 1.f + eps) - std::log(t_m + 1.f + eps);

        // 로스에 반영
        loss += 0.5f * _clog * _clog;

        // dL/dm (로그 차 항에 대한 미분)
        float grad_log = _clog / (c_m + 1.f + eps);
        grid.nodes[i][j][k].dLdm = grad_log;

        // -------------------
        // (2) 소프트 최소질량 벌점
        // -------------------
        // c_m이 min_mass 이하라면, (min_mass - c_m)^2 형태의 추가 로스 부과
        if(c_m < min_mass) {
            // (a) 벌점 계산
            float diff          = min_mass - c_m;
            float penalty_term  = penalty_weight * diff * diff;

            // 로스에 추가
            loss += penalty_term;

            // (b) 벌점의 gradient (d/dm)
            // penalty_term = w * (min_mass - m)^2
            //              = w * (diff^2)
            // d/dm = 2 * w * (diff) * (-1)
            //       = -2 * w * diff
            float grad_penalty = -2.f * penalty_weight * diff;

            // 기존 dLdm에 누적
            grid.nodes[i][j][k].dLdm += grad_penalty;
        }
    }

    // -----------------------------------------
    // 아래는 PointCloud 쪽 위치(x) 변화에 대한 dL/dx 계산
    // -----------------------------------------
#pragma omp parallel for
    for (int p = 0; p < point_cloud.points.size(); p++) {
        MaterialPoint& __restrict mp = point_cloud.points[p];
        mp.dLdx.setZero();

        std::vector<std::array<int, 3>> indices;
        auto nodes = grid.QueryPoint_CubicBSpline(mp.x, &indices);

        for (int i = 0; i < (int)nodes.size(); i++) {
            const GridNode& node = nodes[i];

            Vec3 xg = node.x;
            Vec3 xp = mp.x;
            Vec3 dgp = xg - xp;

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

            // target mass가 0이 아닌 곳은 일반 dLdm
            if (fabs(target_grid->nodes[indices[i][0]][indices[i][1]][indices[i][2]].m) > std::numeric_limits<float>::epsilon()) {
                mp.dLdx += mp.m * node.dLdm * wgpGrad;
            }
            else {
                // target mass가 거의 0이면 out_of_target_penalty 가중치
                mp.dLdx += out_of_target_penalty * mp.m * node.dLdm * wgpGrad;
            }
        }

        // 나머지 항은 여기서는 0으로 처리
        mp.dLdF.setZero();
        mp.dLdv.setZero();
        mp.dLdC.setZero();
    }

    return loss;
}

void DiffMPMLib3D::CompGraph::ComputeForwardPass(size_t start_layer)
{
	for (size_t i = start_layer; i < layers.size() - 1; i++)
	{
		SingleThreadMPM::ForwardTimeStep(
			*layers[i + 1].point_cloud,
			*layers[i].point_cloud,
			*layers[i].grid,
			smoothing_factor, dt, drag, f_ext);
	}
}

void DiffMPMLib3D::CompGraph::ComputeBackwardPass(size_t control_layer)
{
	for (int i = (int)layers.size() - 2; i >= (int)control_layer; i--)
	{
		layers[i].grid->ResetGradients();
		layers[i].point_cloud->ResetGradients();

		Back_Timestep(layers[i + 1], layers[i], drag, dt, smoothing_factor);
	}
}

DiffMPMLib3D::CompGraph::CompGraph(std::shared_ptr<PointCloud> initial_point_cloud, std::shared_ptr<Grid> grid, std::shared_ptr<const Grid> _target_grid)
{
	layers.clear();
	layers.resize(1);

	layers[0].point_cloud = std::make_shared<PointCloud>(*initial_point_cloud);
	layers[0].grid = std::make_shared<Grid>(*grid);

	target_grid = _target_grid;
}

void DiffMPMLib3D::CompGraph::OptimizeDefGradControlSequence(
	// SIMULATION
	int num_steps, // number of floates, aka layers in the comp graph
	float _dt,
	float _drag,
	Vec3 _f_ext,

	// OPTIMIZATION
	int control_stride, // stride between control frames
	int max_gd_iters, // max gradient descent iterations per control frame
	int max_line_search_iters, // max line search iterations per control frame
	float initial_alpha, // initial gradient descent step size
	float gd_tol, // tolerance factor used for determining when gradient descent has converged
	float _smoothing_factor
)
{
	std::streamsize prev_precision = std::cout.precision(6);
	dt = _dt;
	drag = _drag;
	f_ext = _f_ext;
	smoothing_factor = _smoothing_factor;

	std::ios_base::sync_with_stdio(false);

	std::cin.tie(NULL); 
	std::cout.tie(NULL);

	std::cout << "Optimizing control sequence with simulation parameters: " << std::endl;
	std::cout << "num_steps = " << num_steps << std::endl;
	std::cout << "dt = " << dt << std::endl;
	std::cout << "drag = " << drag << std::endl;
	std::cout << "f_ext = " << f_ext.transpose() << std::endl;
	std::cout << "smoothing_factor = " << smoothing_factor << std::endl;

#pragma omp parallel
	{
#pragma omp master
		{
			int num_threads = omp_get_num_threads();
			std::cout << "Number of threads in use: " << num_threads << std::endl;
		}
	}

	// Initialize the computation graph
	SetUpCompGraph(num_steps);

	// Compute initial forward pass and backward pass
#pragma omp parallel
	{
#pragma omp single
		{
			ComputeForwardPass(0);
		}
	}
	float initial_loss = EndLayerMassLoss();
	std::cout << "initial loss = " << initial_loss << std::endl;
	
	ComputeBackwardPass(0);
	float initial_norm = layers.front().point_cloud->Compute_dLdF_Norm();
	std::cout << "initial norm = " << initial_norm << std::endl;

	float convergence_norm = gd_tol * initial_norm; // if norm is less than this, we say the float has converged
	std::cout << "convergence norm = " << convergence_norm << std::endl;

	int totalTemporalIterations = 5;

	float beta1 = 0.9f; // Exponential decay rate for the first moment estimates
	float beta2 = 0.999f; // Exponential decay rate for the second moment estimates
	float epsilon = 1e-5f; // Small constant to prevent division by zero
	float gradient_norm = 0.f;

	for (int temporalIter = 0; temporalIter < totalTemporalIterations; temporalIter++) {
		for (int control_float = 0; control_float < (int)layers.size() - 1; control_float += control_stride) {

			std::cout << "Optimizing for float: " << control_float << std::endl;

			float alpha = initial_alpha;

			for (int gd_iter = 0; gd_iter < max_gd_iters; gd_iter++) {

				std::cout << "gradient descent iteration = " << gd_iter << std::endl;

				float gd_loss = 0.f;

				// 1. compute forward pass from the control float
				ComputeForwardPass(control_float);

				// 2. get the current loss and final layer loss gradients
				gd_loss = EndLayerMassLoss();

				std::cout << "gd loss = " << gd_loss << std::endl;

				// 3. propagate loss gradients to control float
				ComputeBackwardPass(control_float);

				// 4. get the gradient norm
				gradient_norm = layers.front().point_cloud->Compute_dLdF_Norm();
				gradientNormHistory.push_back(gradient_norm);
				

				// 5. Check if we have converged
				if (gradient_norm < gd_tol * initial_norm) {
					std::cout << "gradient norm = " << gradient_norm << " < " << convergence_norm << std::endl;
					std::cout << "control float " << control_float << " converged. Exiting gradient descent and moving to next control float." << std::endl;
					break;
				}

				// 6. Line search to determine a step size where the loss function decreases
				float ls_loss = gd_loss;
				bool ls_loss_decrease_found = false;

				for (int ls_iter = 0; ls_iter < max_line_search_iters; ls_iter++) {
					layers[control_float].point_cloud->Descend_Adam(alpha, gradient_norm, beta1, beta2, epsilon, timestep);
					//layers[control_float].point_cloud->Descend_AMSGrad(alpha, gradient_norm, beta1, beta2, epsilon, timestep);
					layers[control_float].point_cloud->Descend_Lion(alpha, beta1, gradient_norm);
					//layers[control_float].point_cloud->Descend_GradientDescent(alpha, gradient_norm);

					ComputeForwardPass(control_float);
					ls_loss = EndLayerMassLoss();

					if (ls_loss < gd_loss) {
						ls_loss_decrease_found = true;
						std::cout << "line search decrease found at ls_iter = " << ls_iter << ", alpha = " << alpha << ", loss = " << ls_loss << std::endl;
						break;
					}

					layers[control_float].point_cloud->Descend_Adam(-alpha, gradient_norm, beta1, beta2, epsilon, timestep);
					//layers[control_float].point_cloud->Descend_AMSGrad(-alpha, gradient_norm, beta1, beta2, epsilon, timestep);
					layers[control_float].point_cloud->Descend_Lion(-alpha, beta1, gradient_norm);
					//layers[control_float].point_cloud->Descend_GradientDescent(-alpha, gradient_norm);

					alpha /= 2.f;
				}

				if (!ls_loss_decrease_found) {	
					std::cout << "Line search unable to find a decrease in float (" << control_float <<
						"), gd iteration (" << gd_iter << ")" << std::endl;
					std::cout << "Exiting gradient descent and moving to next control float." << std::endl;
					break;
				}

				timestep += 1;
			}
		}
	}

	float final_loss = EndLayerMassLoss();
	std::cout << "Initial loss = " << initial_loss << std::endl;
	std::cout << "Final loss = " << final_loss << std::endl;

	gradient_norm = layers.front().point_cloud->Compute_dLdF_Norm();

	if (final_loss >= initial_loss) {
		std::cout << "Unable to find a loss decrease." << std::endl;
	}

	std::cout << "Final Gradient Norm: " << gradient_norm << std::endl;
	std::cout << "Deformation gradient control sequence optimization finished." << std::endl;

	{
		std::ofstream ofs("gradient_norm_log.csv");
		if (ofs.is_open()) {
			// 형식: 한 줄에 하나씩 Gradient Norm만 기록
			// 원하면 인덱스나 gd_loss도 같이 적을 수 있음
			for (size_t i = 0; i < gradientNormHistory.size(); i++) {
				ofs << i << "," << gradientNormHistory[i] << "\n";
			}
			ofs.close();
		}
		else {
			std::cerr << "Failed to open gradient_norm_log.csv for writing.\n";
		}
	}
	std::cout.precision(prev_precision);
}


void DiffMPMLib3D::CompGraph::FiniteDifferencesGradientTest(int num_steps, size_t particle_id)
{
	/*******FINITE DIFFERENCES TEST********/
	std::streamsize prev_precision = std::cout.precision(6);

	std::cout << "FINITE DIFFERENCES TEST: dLdF" << std::endl;
	SetUpCompGraph(num_steps);
	ComputeForwardPass(0);
	float initial_loss = EndLayerMassLoss();
	std::cout << "initial loss = " << initial_loss << std::endl;
	ComputeBackwardPass(0);


	MaterialPoint& __restrict mp = layers.front().point_cloud->points[particle_id];

	std::cout << "analytic dLdF for particle 0:\n" << mp.dLdF << std::endl;


	Mat3 fd_dLdF;
	float delta = 1e-9f;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			//MPMForwardSimulation(stcg, f_ext, drag, dt, control_float, false);
			float originalValue = mp.F(i, j);

			mp.F(i, j) = originalValue + delta;
			ComputeForwardPass(0);
			float l1 = EndLayerMassLoss();

			mp.F(i, j) = originalValue - delta;
			ComputeForwardPass(0);
			float l2 = EndLayerMassLoss();

			fd_dLdF(i, j) = (l1 - l2) / (2.f * delta);
			mp.F(i, j) = originalValue;
		}
	}
	std::cout << "finite differences dLdF for particle " << particle_id << ":\n" << fd_dLdF << std::endl;

	float grad_diff = (fd_dLdF - mp.dLdF).norm();
	std::cout << "gradient difference = " << grad_diff << std::endl;

	float grad_diff_percent = 100.f * grad_diff / fd_dLdF.norm();
	std::cout << "gradient difference percentage = " << grad_diff_percent << std::endl;

	std::cout.precision(prev_precision);
}
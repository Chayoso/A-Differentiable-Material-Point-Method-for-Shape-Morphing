#include "pch.h"
#include "PointCloud.h"
#include "Elasticity.h"
#include <fstream>
#include <random>
#include <iostream>
#include <chrono>

template <typename ForwardIt, typename T>
void custom_iota(ForwardIt first, ForwardIt last, T value) {
	while (first != last) {
		*first++ = value++;
	}
}

void DiffMPMLib3D::PointCloud::ResetGradients()
{
#pragma omp parallel for
	for (int p = 0; p < points.size(); p++) {
		points[p].ResetGradients();
	}
}

float DiffMPMLib3D::PointCloud::Compute_dLdF_Norm()
{
	float norm = 0;

#pragma omp parallel for reduction(+:norm)
	for (int i = 0; i < points.size(); ++i) {
		norm += points[i].dLdF.squaredNorm();
	}

	norm = sqrt(norm);
	return norm;
}

std::vector<int> DiffMPMLib3D::PointCloud::generateMinibatchIndices(int total_size, int minibatch_size)
{
	std::vector<int> indices(total_size);
	custom_iota(indices.begin(), indices.end(), 0); 
	std::random_device rd;
	std::mt19937 g(rd());
	g.seed(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())); // 시드 설정
	std::shuffle(indices.begin(), indices.end(), g);
	indices.resize(minibatch_size);

	return indices;
}


float DiffMPMLib3D::PointCloud::Compute_dLdF_Norm_Stochastic(const std::vector<int>& minibatch_indices)
{
	float norm = 0;

#pragma omp parallel for reduction(+:norm)
	for (int idx = 0; idx < minibatch_indices.size(); ++idx) {
		int i = minibatch_indices[idx];
		norm += points[i].dLdF.squaredNorm();
	}

	norm = std::sqrt(norm);
	return norm;
}

void DiffMPMLib3D::PointCloud::Descend_Adam_Stochastic(float alpha, float gradient_norm, float beta1, 
	float beta2, float epsilon, int timestep, const std::vector<int>& minibatch_indices)
{
#pragma omp parallel for
	for (int idx = 0; idx < minibatch_indices.size(); ++idx) {
		int i = minibatch_indices[idx];
		Mat3 grad = points[i].dLdF / gradient_norm;

		points[i].momentum = beta1 * points[i].momentum + (1.0f - beta1) * grad;
		points[i].vector = beta2 * points[i].vector + (1.0f - beta2) * (grad.array().square().matrix());

		Mat3 m_hat = points[i].momentum / (1.0f - std::pow(beta1, timestep + 1));
		Mat3 v_hat = points[i].vector / (1.0f - std::pow(beta2, timestep + 1));

		points[i].dFc -= alpha * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
	}
}

void DiffMPMLib3D::PointCloud::Descend_Adam(float alpha, float gradient_norm, float beta1, float beta2, float epsilon, int timestep)
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		Mat3 grad = points[i].dLdF / gradient_norm;

		points[i].momentum = beta1 * points[i].momentum + (1.0f - beta1) * grad;

		points[i].vector = beta2 * points[i].vector + (1.0f - beta2) * (grad.array().square().matrix());

		Mat3 m_hat = points[i].momentum / (1.0f - pow(beta1, timestep));
		Mat3 v_hat = points[i].vector / (1.0f - pow(beta2, timestep));

		points[i].dFc -= alpha * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
	}
}

void DiffMPMLib3D::PointCloud::Descend_AMSGrad(
	float alpha, float gradient_norm,
	float beta1, float beta2, float epsilon, int timestep)
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) Gradient (scaled by gradient_norm if needed)
		Mat3 grad = points[i].dLdF / gradient_norm;

		// 2) m, v 업데이트 (Adam)
		points[i].momentum = beta1 * points[i].momentum + (1.f - beta1) * grad;  // m_t
		points[i].vector = beta2 * points[i].vector
			+ (1.f - beta2) * (grad.array().square().matrix());  // v_t

		// 3) 바이어스 보정
		Mat3 m_hat = points[i].momentum / (1.f - std::pow(beta1, timestep));
		Mat3 v_hat = points[i].vector / (1.f - std::pow(beta2, timestep));

		// 4) AMSGrad: v_max 갱신
		points[i].vector_max = points[i].vector_max.cwiseMax(v_hat);

		// 5) 파라미터(dFc) 업데이트
		// 올바른 예시
		Mat3 update = (
			m_hat.array()
			/ (points[i].vector_max.array().sqrt() + epsilon)
			).matrix();

		points[i].dFc -= alpha * update;
	}
}

void DiffMPMLib3D::PointCloud::Descend_Lion(
	float alpha,   // 학습률
	float beta,    // 모멘텀 계수
	float gradient_norm)
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) grad 가져오기 (원한다면 gradient_norm으로 스케일링 가능)
		//    Lion 논문 자체는 gradient_norm을 안 쓰지만, 필요하면 적용.
		Eigen::Matrix3f grad = points[i].dLdF / gradient_norm;

		// 2) sign(grad) 계산
		Eigen::Matrix3f signGrad = grad.unaryExpr(
			[this](float x) {
				return SignFloat(x);
			}
		);

		// 3) Lion 모멘텀 업데이트
		//    m_t = beta * m_{t-1} + (1-beta) * sign(grad)
		points[i].momentum = beta * points[i].momentum
			+ (1.f - beta) * signGrad;

		// 4) 파라미터(dFc) 업데이트
		//    theta <- theta - alpha * m_t
		points[i].dFc -= alpha * points[i].momentum;
	}
}

void DiffMPMLib3D::PointCloud::Descend_GradientDescent(
	float alpha,         // 학습률(스텝 크기)
	float gradient_norm) // 필요하다면 gradient_norm으로 스케일링
{
#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		// 1) 기울기 가져오기 (dLdF), 스케일링
		//    (gradient_norm을 1로 두면 그냥 raw gradient 사용)
		Eigen::Matrix3f grad = points[i].dLdF / gradient_norm;

		// 2) GD update: theta <- theta - alpha * grad
		points[i].dFc -= alpha * grad;
	}
}


void DiffMPMLib3D::PointCloud::Descend_dLdF(float alpha, float gradient_norm)
{
	float momentum_factor = 0.9f;

	float min_clip_value = 1e-3f; 
	float max_clip_value = 1.0f;  

#pragma omp parallel for
	for (int i = 0; i < points.size(); ++i)
	{
		Mat3 dir = points[i].dLdF / gradient_norm;

		// Clipping Gradient;
		dir = dir.unaryExpr([=](float val) {
			if (val > max_clip_value) return max_clip_value;
			if (val < -max_clip_value) return -max_clip_value;
			if (std::abs(val) < min_clip_value) return (val < 0 ? -min_clip_value : min_clip_value);
			return val;
			});
	
		points[i].momentum = momentum_factor * points[i].momentum + alpha * dir;
		points[i].dFc -= points[i].momentum;
	}
}

void DiffMPMLib3D::PointCloud::RemovePoint(size_t point_index)
{
	points.erase(points.begin() + point_index);
}

bool DiffMPMLib3D::PointCloud::ReadFromOBJ(std::string obj_path, float point_mass)
{
	// only material parameter assigned will be mass, since we only use this for rendering data (not simulation)

	points.resize(0);


	std::ifstream ifs;
	ifs.open(obj_path);
	if (ifs.good()) {
		std::string junk;
		float x, y, z;

		while (ifs >> junk >> x >> y >> z)
		{
			Eigen::Vector3f pos(x, y, z);
			MaterialPoint point;
			point.x = pos;
			point.m = point_mass;
			points.emplace_back(point);
		}
		ifs.close();
		return true;
	}
	else {
		std::cout << "couldn't open file: " << obj_path << std::endl;
		ifs.close();
		return false;
	}
}

void DiffMPMLib3D::PointCloud::WriteToOBJ(std::string obj_path)
{
	std::ofstream ofs(obj_path, std::ios::out);

	if (!ofs) {
		std::cerr << "Could not open file for writing: " << obj_path << std::endl;
		return;
	}

	for (size_t v = 0; v < points.size(); v++) {
		ofs << "v " << points[v].x[0] << " " << points[v].x[1] << " " << points[v].x[2] << "\n";
	}

	ofs.close();
}

void DiffMPMLib3D::PointCloud::WriteMassVelocityDefgradsToFile(std::string file_path)
{
	std::ofstream ofs;
	ofs.open(file_path);

	if (ofs.good()) {
		for (size_t v = 0; v < points.size(); v++) {
			ofs << "x " << points[v].x[0] << " " << points[v].x[1] << " " << points[v].x[2] << "\n";
			ofs << "v " << points[v].v[0] << " " << points[v].v[1] << " " << points[v].v[2] << "\n";


			// I don't plan on doing any visualizations that actually use the 3x3 deformation gradients,
			// just maybe their elastic energy and their magnitudes?

			ofs << "F";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].F(i);
			}
			ofs << "\n";
			ofs << "dFc";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].dFc(i);
			}
			ofs << "\n";
			ofs << "dLdF";
			for (size_t i = 0; i < 9; i++) {
				ofs << " " << points[v].dLdF(i);
			}
			ofs << "\n";
		}
	}
	ofs.close();
}

//void DiffMPMLib3D::PointCloud::WriteEntirePointCloudToFile(std::string file_path)
//{
//	// not sure how much I want to invest into this codebase in the future, so just doing the easy thing here
//	std::ofstream ofs;
//	ofs.open(file_path);
//
//	if (ofs.good()) {
//		// first line: number of points
//		ofs.precision(17);
//		ofs << points.size() << std::endl;
//		for (size_t v = 0; v < points.size(); v++) {
//			points[v].WriteEntirePointToFile(ofs);
//		}
//	}
//	ofs.close();
//}
//
//bool DiffMPMLib3D::PointCloud::ReadEntirePointCloudFromFile(std::string file_path)
//{
//	std::ifstream ifs;
//	ifs.open(file_path);
//
//	if (ifs.good()) {
//		// first line: number of points
//		size_t num_points;
//		ifs >> num_points;
//
//		points.resize(num_points);
//
//		for (size_t v = 0; v < num_points; v++) {
//			points[v].ReadEntirePointFromFile(ifs);
//		}
//
//		ifs.close();
//		return true;
//	}
//	else {
//		std::cout << "couldn't read file: " << file_path << std::endl;
//		return false;
//	}
//}

void DiffMPMLib3D::PointCloud::WriteEntirePointCloudToBinaryFile(std::string file_path)
{
	std::ofstream ofs;
	ofs.open(file_path, std::ios::binary);

	if (ofs.good()) {
		cereal::BinaryOutputArchive oarchive(ofs); // Create an output archive

		oarchive(*this);
	}
	ofs.close();
}

bool DiffMPMLib3D::PointCloud::ReadEntirePointCloudFromBinaryFile(std::string file_path)
{
	std::ifstream ifs;
	ifs.open(file_path, std::ios::binary);
	if (ifs.good()) {
		cereal::BinaryInputArchive iarchive(ifs); // Create an input archive

		iarchive(*this);
	}
	else {
		std::cout << "couldn't read file: " << file_path << std::endl;
		return false;
	}
	ifs.close();
	return true;
}

bool DiffMPMLib3D::PointCloud::IsEqualToOtherPointCloud(const PointCloud& other_pc)
{
	for (int i = 0; i < points.size(); i++) {
		const MaterialPoint& mp = points[i];
		const MaterialPoint& other_mp = other_pc.points[i];
		if (!mp.IsEqualToOtherPoint(other_mp)) {

			std::cout << "Not equal at point " << i << std::endl;
			std::cout << "*** original MP ***" << std::endl;
			mp.PrintMP();
			std::cout << "*** other MP ***" << std::endl;
			other_mp.PrintMP();
			return false;
		}
	}

	std::cout << "Point clouds are equal" << std::endl;
	return true;
}


std::vector<DiffMPMLib3D::Vec3> DiffMPMLib3D::PointCloud::GetPointPositions() const
{
	std::vector<Vec3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].x;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointMasses() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].m;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointVolumes() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].vol;
	}

	return ret;
}

std::vector<DiffMPMLib3D::Mat3> DiffMPMLib3D::PointCloud::GetPointDefGrads() const
{
	std::vector<Mat3> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = points[p].F;
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointDeterminants() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = (points[p].F + points[p].dFc).determinant();
	}

	return ret;
}

std::vector<float> DiffMPMLib3D::PointCloud::GetPointElasticEnergies() const
{
	std::vector<float> ret;
	ret.resize(points.size());

#pragma omp parallel for
	for (int p = 0; p < points.size(); p++)
	{
		ret[p] = FixedCorotatedElasticity(points[p].F + points[p].dFc, points[p].lam, points[p].mu);
	}

	return ret;
}

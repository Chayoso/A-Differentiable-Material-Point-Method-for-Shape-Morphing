#pragma once
#include "pch.h"
#include "Tensor3x3x3x3.h"

namespace DiffMPMLib3D {
	Tensor3x3x3x3 d_JFit_dF_FD(const Mat3& F);

	float FixedCorotatedElasticity(const Mat3& F, float lam, float mu);
	Mat3 PK_FixedCorotatedElasticity(const Mat3& F, float lam, float mu);

	Mat3 d2_FCE_psi_dF2_mult_by_dF(const Mat3& F, float lam, float mu, const Mat3& dF);




	Tensor3x3x3x3 d2_FCE_psi_dF2_FD(const Mat3& F, float lam, float mu);

	Tensor3x3x3x3 d2_FCE_psi_dF2_mult_trick(const Mat3& F, float lam, float mu);

	Tensor3x3x3x3 d_JFit_dF(float J, const Mat3& Fit);


	void CalculateLameParameters(float young_mod, float poisson, float& lam, float& mu);
}
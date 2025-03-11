#include "pch.h"
#include "Elasticity.h"
#include "qrsvd/ImplicitQRSVD.h"
#include <chrono>
#include <unsupported/Eigen/KroneckerProduct>

DiffMPMLib3D::Tensor3x3x3x3 DiffMPMLib3D::d_JFit_dF_FD(const Mat3& F)
{
    Tensor3x3x3x3 ret;

    auto CalcJFit = [](const Mat3& _F)
        {
            float J = _F.determinant();
            Mat3 Fit = _F.inverse().transpose();
            Mat3 ret = J * Fit;
            return ret;
        };

    Mat3 temp_F = F;
    float delta = 1e-6;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float originalValue = temp_F(i, j);

            temp_F(i, j) = originalValue + delta;
            Mat3 JFit_forward = CalcJFit(temp_F);
            //std::cout << "JFit_forward\n" << JFit_forward << std::endl;

            temp_F(i, j) = originalValue - delta;
            Mat3 JFit_backward = CalcJFit(temp_F);
            //std::cout << "JFit_backward\n" << JFit_backward << std::endl;

            temp_F(i, j) = originalValue;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++)
                {
                    float l1 = JFit_forward(a, b);
                    float l2 = JFit_backward(a, b);
                    ret[a][b](i, j) = (l1 - l2) / (2.0 * delta);
                }
            }
        }
    }


    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float originalValue = temp_F(i, j);

            temp_F(i, j) = originalValue + delta;
            Mat3 JFit_forward = CalcJFit(temp_F);
            //std::cout << "JFit_forward\n" << JFit_forward << std::endl;

            temp_F(i, j) = originalValue - delta;
            Mat3 JFit_backward = CalcJFit(temp_F);
            //std::cout << "JFit_backward\n" << JFit_backward << std::endl;

            temp_F(i, j) = originalValue;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++)
                {
                    float l1 = JFit_forward(a, b);
                    float l2 = JFit_backward(a, b);
                    ret[a][b](i, j) = (l1 - l2) / (2.0 * delta);
                }
            }
        }
    }

    return ret;
}

float DiffMPMLib3D::FixedCorotatedElasticity(const Mat3& F, float lam, float mu)
{
    Mat3 U, V;
    Vec3 S;
    JIXIE::singularValueDecomposition(F, U, S, V);

    float svd_sum = (S[0] - 1.0) * (S[0] - 1.0) + (S[1] - 1.0) * (S[1] - 1.0) + (S[2] - 1.0) * (S[2] - 1.0);
    float J = F.determinant();

    J = std::max(J, 1e-6f);

    return mu * svd_sum + 0.5f * lam * (J - 1.0) * (J - 1.0);
}

DiffMPMLib3D::Mat3 DiffMPMLib3D::PK_FixedCorotatedElasticity(const Mat3& F, float lam, float mu)
{
    Mat3 R, S;
    JIXIE::polarDecomposition(F, R, S);

    float J = F.determinant();

    J = std::max(J, 1e-6f);

    return 2.f * mu * (F - R) + lam * (J - 1.f) * J * F.inverse().transpose();
}

DiffMPMLib3D::Mat3 DiffMPMLib3D::d2_FCE_psi_dF2_mult_by_dF(const Mat3& F, float lam, float mu, const Mat3& dF)
{

    Mat3 R, S;
    JIXIE::polarDecomposition(F, R, S);
    float J = F.determinant();
    Mat3 Finv = F.inverse();
    Mat3 Fit = Finv.transpose();

    J = std::max(J, 1e-6f);

    Mat3 ret = Mat3::Zero();

    // 2 mu dF 
    ret += 2.f * mu * dF;

    // + lam J F^-T (J F^-T : dF)
    ret += lam * J * Fit * InnerProduct(J * Fit, dF);

    Tensor3x3x3x3 dJFit_dF = d_JFit_dF_FD(F);

    Mat3 term = Mat3::Zero();

#pragma omp parallel for schedule(guided)
    for (int idx = 0; idx < 9; idx++) {
        int a = idx / 3;
        int b = idx % 3;

        term(a, b) = 0.0;
        for (int ij = 0; ij < 9; ij++) {
            int i = ij / 3;
            int j = ij % 3;

#pragma omp atomic
            term(a, b) += dJFit_dF[a][b](i, j) * dF(i, j);
        }
    }
    ret += lam * (J - 1.f) * term;

    // Analytical
    //ret += lam * (J - 1.0) * dF.adjoint().transpose();

    // Now time to calculate dR
    // Derivation done in sympy
    float s00, s01, s02, s11, s12, s22;
    s00 = S(0, 0);
    s01 = S(0, 1);
    s02 = S(0, 2);
    s11 = S(1, 1);
    s12 = S(1, 2);
    s22 = S(2, 2);
    Mat3 A;
    A.row(0) = Vec3(s02, s12, -(s00 + s11));
    A.row(1) = Vec3(-s01, s00 + s22, -s12);
    A.row(2) = Vec3(-(s11 + s22), s01, s02);

    float A_condition_number = A.fullPivHouseholderQr().logAbsDeterminant();
    if (A_condition_number > 20.0) {
        A += Mat3::Identity() * 1e-4f;
    }
    
    Mat3 temp = R.transpose() * dF - dF.transpose() * R;
    Vec3 b = Vec3(temp(0, 1), temp(0, 2), temp(1, 2));

    // Let x = R.T * dR
    // solve Ax = b for x
    Vec3 x = A.colPivHouseholderQr().solve(b);

    // dR = R(R.T * dR)
    Mat3 dR;
    dR.row(0) = Vec3(0.f, -x(2), x(1));
    dR.row(1) = Vec3(x(2), 0.f, -x(0));
    dR.row(2) = Vec3(-x(1), x(0), 0.f);
    dR = R * dR;

    ret -= 2.f * mu * dR;

    return ret;
}


DiffMPMLib3D::Tensor3x3x3x3 DiffMPMLib3D::d2_FCE_psi_dF2_FD(const Mat3& F, float lam, float mu)
{
    // Return tensor of indices: a, b, i, j,  for dP[a][b] / dF[i][j]

    Tensor3x3x3x3 ret;

    float delta = 1e-6f;
    Mat3 temp_F = F;
    Mat3 P = PK_FixedCorotatedElasticity(temp_F, lam, mu);

#pragma omp parallel for collapse(4) 
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float originalValue = temp_F(i, j);

            temp_F(i, j) = originalValue + delta;
            Mat3 P_forward = PK_FixedCorotatedElasticity(temp_F, lam, mu);

            temp_F(i, j) = originalValue - delta;
            Mat3 P_backward = PK_FixedCorotatedElasticity(temp_F, lam, mu);

            temp_F(i, j) = originalValue;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++)
                {
                    float l1 = P_forward(a, b);
                    float l2 = P_backward(a, b);
                    ret[a][b](i, j) = (l1 - l2) / (2.f * delta);
                }
            }
        }
    }

    return ret;
}

DiffMPMLib3D::Tensor3x3x3x3 DiffMPMLib3D::d2_FCE_psi_dF2_mult_trick(const Mat3& F, float lam, float mu)
{
    Tensor3x3x3x3 ret;
    Mat3 E;// = Mat3::Identity();
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            E.setZero();
            E(a, b) = 1.f;
            ret[a][b] = d2_FCE_psi_dF2_mult_by_dF(F, lam, mu, E);
        }
    }
    return ret;
}

void DiffMPMLib3D::CalculateLameParameters(float young_mod, float poisson, float& lam, float& mu)
{
    lam = young_mod * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
    mu = young_mod / (2.f + 2.f * poisson);
}

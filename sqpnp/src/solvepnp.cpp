#include "sqpnp/solvepnp.h"

#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Dense>

#include "solvesqp.h"

namespace Eigen
{
	typedef Eigen::Matrix<double, 9, 1> Vector9d;
	typedef Eigen::Matrix<double, 9, 9> Matrix9d;
}

Eigen::Vector9d
get_r(const unsigned int mu, const Eigen::Vector9d& e)
{
	Eigen::MatrixXd reshaped_e = std::pow(-1, mu) * sqrt(3) * e;
	reshaped_e.resize(3, 3);
	reshaped_e.transposeInPlace();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(reshaped_e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd r_mat = svd.matrixU() * svd.matrixV().transpose();
	if (r_mat.determinant() < 0)
	{
		Eigen::MatrixXd u = svd.matrixU();
		u.block<1, 3>(2, 0) *= -1;
		r_mat = u * svd.matrixV().transpose();
	}
	r_mat.transposeInPlace();
	r_mat.resize(9, 1);
	return Eigen::Vector9d(r_mat);
}

bool
solvePnP(const std::vector<Eigen::Vector2d>& image_points,
	const std::vector<Eigen::Vector3d>& world_points,
	const double tolerance,
	const unsigned int max_iterations)
{
	if (image_points.size() != world_points.size() || image_points.size() < 3) return false;
	if (tolerance > 10e-5 || max_iterations < 15) return false;

	size_t n = image_points.size();

	std::vector<Eigen::Matrix<double, 3, 9>> A;
	A.resize(n);

	std::vector<Eigen::Matrix3d> Q;
	Q.resize(n);


	Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
	Eigen::MatrixXd QA_sum = Eigen::MatrixXd::Zero(3, 9);

	//Compute A and Q
	for (size_t i = 0; i < n; ++i)
	{
		A[i] = Eigen::MatrixXd::Zero(3, 9);
		A[i].block<1, 3>(0, 0) = world_points[i];
		A[i].block<1, 3>(1, 3) = world_points[i];
		A[i].block<1, 3>(2, 6) = world_points[i];

		Q[i] << 1, 0, -image_points[i][0],
			0, 1, -image_points[i][1],
			-image_points[i][0], -image_points[i][1], (image_points[i][0] * image_points[i][0] + image_points[i][1] * image_points[i][1]);

		Q_sum += Q[i];
		QA_sum += Q[i] * A[i];
	}

	Eigen::MatrixXd P = -1 * Q_sum.inverse() * QA_sum;

	Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(9, 9);
	for (size_t i = 0; i < n; ++i)
	{
		Omega += ((A[i] + P).transpose() * Q[i] * (A[i] + P));
	}

	A.clear();
	Q.clear();

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Omega, Eigen::ComputeThinU);
	size_t k = std::max(9 - svd.rank(), long long(1));
	Eigen::VectorXd s = svd.singularValues();
	Eigen::MatrixXd e = -1*svd.matrixU();

	std::cout << s << std::endl;

	std::cout << "Size of nullspace: " <<  k << std::endl;

	//Assign nullspace vectors according to:
	// argmin x^T Omega x^ for x in S^8
	// how to assign this?

	std::vector<std::pair<double, Eigen::VectorXd>> r_est(18);

	std::cout << "i,\tk,\tnu" << std::endl;
	for (unsigned int i = 1; i <= 2 * k; i++)
	{
		unsigned int idx = i - 1;
		double mu = floor((double)(idx) / k);
		unsigned int nu = 9 - k + i - floor((double)(idx) / k) * k;
		Eigen::Vector9d e_nu = e.block<9, 1>(0, nu - 1);
		std::cout << i << ",\t" << k << "\t," << nu << std::endl;

		auto r = get_r(mu, e_nu);
		r_est[idx].second = solveSQP(r, Omega, tolerance, max_iterations);
		r_est[idx].first = r_est[idx].second.transpose() * Omega * r_est[idx].second;
	}

	auto min_error = [&]() -> double
	{
		double min_err = std::numeric_limits<double>::max();
		for (unsigned int i = 0; i < 2*k; ++i)
		{
			if (r_est[i].first < min_err) min_err = r_est[i].first;
		}
		return min_err;
	};

	size_t k_tmp = k;

	/*while (min_error() >= s[9 - k_tmp])
	{
		for (size_t i = 0; i <= 1; ++i)
		{
			auto r = get_r(i + 1, e.block<9, 1>(0, 9 - k_tmp));
			size_t idx = size_t(2) * k_tmp + i - 3;
			std::cout << i << ", " << k_tmp << ", " << idx << std::endl;
			r_est[idx].second = solveSQP(r, Omega, tolerance, max_iterations);
			r_est[idx].first = r_est[idx].second.transpose() * Omega * r_est[idx].second;
		}
		++k_tmp;
		std::cout << min_error() << ", " << s[9 - k_tmp] << std::endl;
	}*/

	std::sort(r_est.begin(), r_est.begin()+2*k, [](auto& left, auto& right)
	{
		return left.first < right.first;
	});
	std::cout << "Finished: " << k << std::endl;
	for (int i = 0; i < 2*k; i++)
	{

		std::cout << i << " : " << r_est[i].first << " : " << r_est[i].second.transpose() << std::endl;
	}



	return true;
}
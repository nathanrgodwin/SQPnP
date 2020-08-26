#include "sqpnp/solvepnp.h"

#include <Eigen/SVD>

#include "solvesqp.h"

bool
solvePnP(const std::vector<Eigen::Vector2d>& image_points,
	const std::vector<Eigen::Vector3d>& world_points,
	const double tolerance,
	const unsigned int max_iterations)
{
	if (image_points.size() != world_points.size() || image_points.size() < 3) return false;
	if (tolerance > 1e-5 || max_iterations < 15) return false;

	size_t n = image_points.size();

	std::vector<Eigen::MatrixXd> A;
	A.resize(n);

	std::vector<Eigen::Matrix3d> Q;
	Q.resize(n);


	Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
	Eigen::MatrixXd QA_sum = Eigen::MatrixXd::Zero(3, 9);

	//Compute A and Q
	for (size_t i = 0; i < n; ++i)
	{
		A[i] = Eigen::MatrixXd::Zero(3, 9);
		A[i].block<1, 3>(0, 0) = world_points[i];// Eigen::Vector3d(world_points[i][0], world_points[i][1], world_points[i][2]);
		A[i].block<1, 3>(1, 4) = world_points[i];//Eigen::Vector3d(world_points[i][0], world_points[i][1], world_points[i][2]);
		A[i].block<1, 3>(2, 8) = world_points[i];//Eigen::Vector3d(world_points[i][0], world_points[i][1], world_points[i][2]);

		Q[i] << 1, 0, -image_points[i][0],
			0, 1, -image_points[i][1],
			-image_points[i][0], -image_points[i][1], (image_points[i][0] * image_points[i][0] + image_points[i][1] * image_points[i][0]);

		Q_sum += Q[i];
		QA_sum += Q[i] * A[i];
	}

	Eigen::MatrixXd P = -1 * Q_sum.inverse() * QA_sum;


	Eigen::MatrixXd Omega = Eigen::MatrixXd::Zero(9, 9);
	for (size_t i = 0; i < n; ++i)
	{
		Omega += ((A[i] + P).transpose() * Q[i] * (A[i] + P));
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Omega, Eigen::ComputeThinU);
	unsigned int k = 9 - svd.rank();
	Eigen::VectorXd s = svd.singularValues();
	Eigen::MatrixXd e = -1*svd.matrixU();

	//Assign nullspace vectors according to:
	// argmin x^T Omega x^ for x in S^8

	std::vector<Eigen::VectorXd> r_est(2 * k);
	std::vector<double> error(2*k);


	for (unsigned int i = 1; i <= 2 * k; i++)
	{
		unsigned int idx = i - 1;
		double mu = floor(((double)i - 1) / k);
		unsigned int nu = 9 - k + idx - floor((double)idx / k) * k;

		//Assign starting rotations as:
		// r[i] = argmin || x - (-1)^mu sqrt(3) * e_nu || ^2 for mat(x) in SO(3)

		r_est[idx] = solveSQP(r[idx], Omega, tolerance, max_iterations);
		error[idx] = r[idx].transpose() * Omega * r[idx];
	}

	auto min_error = [&]() -> double { return *(std::minmax_element(error.begin(), error.end()).first); };

	while (min_error() >= s[9 - k])
	{
		for (unsigned int i = 1; i <= 2; ++i)
		{
			//r[2*k+i] = argmin || x - (-1)^i * sqrt(3) * e_(9-k) || ^2
			r_est[2 * k + i] = solveSQP(r[2 * k + i], Omega, tolerance, max_iterations);
			error[2 * k + i] = r[2 * k + i].transpose() * Omega * r[2 * k + i];
		}

		++k;
	}

	return true;
}
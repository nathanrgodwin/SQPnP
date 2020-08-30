#include "solvesqp.h"

#include <iostream>

#include <Eigen/Dense>

//For:
//r			9x1
//Omega		9x9
Eigen::VectorXd
solveSQP(const Eigen::VectorXd& r,
	const Eigen::MatrixXd& Omega,
	const double tolerance,
	const unsigned int max_iterations)
{
	Eigen::VectorXd r_est = r;
	unsigned int step = 0;
	Eigen::VectorXd delta_est;
	do
	{
		//Compute h(r)
		Eigen::VectorXd h(6, 1);
		h << r_est[0] * r_est[0] + r_est[1] * r_est[1] + r_est[2] * r_est[2] - 1,
			r_est[3] * r_est[3] + r_est[4] * r_est[4] + r_est[5] * r_est[5] - 1,
			r_est[0] * r_est[3] + r_est[1] * r_est[4] + r_est[2] * r_est[5],
			r_est[0] * r_est[6] + r_est[1] * r_est[7] + r_est[2] * r_est[8],
			r_est[3] * r_est[6] + r_est[4] * r_est[7] + r_est[5] * r_est[8],
			(r_est[0] * r_est[4] * r_est[8]) - (r_est[0] * r_est[5] * r_est[7]) - (r_est[1] * r_est[3] * r_est[8]) + (r_est[1] * r_est[5] * r_est[6]) + (r_est[2] * r_est[3] * r_est[7]) - (r_est[2] * r_est[4] * r_est[6]) - 1;

		//Compute H_r
		Eigen::MatrixXd H_t = Eigen::MatrixXd::Zero(9, 6);
		H_t.block<3, 1>(0, 0) = 2 * r_est.block<3, 1>(0, 0);
		H_t.block<3, 1>(3, 1) = 2 * r_est.block<3, 1>(3, 0);
		H_t.block<3, 1>(0, 2) = r_est.block<3, 1>(3, 0);
		H_t.block<3, 1>(3, 2) = r_est.block<3, 1>(0, 0);
		H_t.block<3, 1>(0, 3) = r_est.block<3, 1>(6, 0);
		H_t.block<3, 1>(6, 3) = r_est.block<3, 1>(0, 0);
		H_t.block<3, 1>(3, 4) = r_est.block<3, 1>(6, 0);
		H_t.block<3, 1>(6, 4) = r_est.block<3, 1>(3, 0);
		
		H_t(0, 5) = r_est[4] * r_est[8] - r_est[5] * r_est[7];
		H_t(1, 5) = r_est[5] * r_est[6] - r_est[3] * r_est[8];
		H_t(2, 5) = r_est[3] * r_est[7] - r_est[4] * r_est[6];
		H_t(3, 5) = r_est[2] * r_est[7] - r_est[1] * r_est[8];
		H_t(4, 5) = r_est[0] * r_est[8] - r_est[2] * r_est[6];
		H_t(5, 5) = r_est[1] * r_est[6] - r_est[0] * r_est[7];
		H_t(6, 5) = r_est[1] * r_est[5] - r_est[2] * r_est[4];
		H_t(7, 5) = r_est[2] * r_est[3] - r_est[0] * r_est[5];
		H_t(8, 5) = r_est[0] * r_est[4] - r_est[1] * r_est[3];

		//Compute delta_est and lambda_est
		Eigen::MatrixXd part1 = Eigen::MatrixXd::Zero(15,15), part2(15,1);
		part1.block<9, 9>(0, 0) = Omega;
		part1.block<6, 9>(9, 0) = H_t.transpose();
		part1.block<6, 6>(9, 9) = Eigen::MatrixXd::Zero(6, 6);
		part1.block<9, 6>(0, 9) = H_t;

		part2.block<9, 1>(0, 0) = -1 * Omega * r_est;
		part2.block<6, 1>(9, 0) = -1 * h;

		Eigen::MatrixXd result = part1.inverse() * part2;

		delta_est = result.block<9, 1>(0, 0);

		//Step r_est
		r_est += delta_est;

		std::cout << delta_est.norm() << " : " << delta_est.transpose() << std::endl;

		//Step counter
		++step;

	} while (delta_est.norm() >= tolerance && step <= max_iterations);
	return r_est;
}
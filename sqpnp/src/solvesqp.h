#include <Eigen/Core>

Eigen::VectorXd
solveSQP(const Eigen::VectorXd& r,
	const Eigen::MatrixXd& Omega,
	const double tolerance,
	const unsigned int max_iterations);
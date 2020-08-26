#include <vector>

#include <Eigen/Core>

bool
solvePnP(const std::vector<Eigen::Vector2d>& image_points,
	const std::vector<Eigen::Vector3d>& world_points,
	const double tolerance,
	const unsigned int max_iterations);
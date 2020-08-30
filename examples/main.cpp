#include <iostream>

#include "sqpnp/solvepnp.h"

int 
main(int argc, char** argv)
{
	std::vector<Eigen::Vector2d> img_pts{
		Eigen::Vector2d(359, 391),
		Eigen::Vector2d(399, 561),
		Eigen::Vector2d(345, 465) }; //Eigen::Vector2d(337, 297) ,//};/*
		/*
		Eigen::Vector2d(345, 465) };/*,
		Eigen::Vector2d(345, 465),
		Eigen::Vector2d(453, 469)
	};//*/
	std::vector<Eigen::Vector3d> world_pts{
		Eigen::Vector3d(0,0,0),
		Eigen::Vector3d(0,-330,-65),
		Eigen::Vector3d(-150,-150,-125) };// Eigen::Vector3d(-225, 170, -135) };
										  /* ,
		Eigen::Vector3d(-150,-150,-125) };/*
		Eigen::Vector3d(-150,-150,-125), 
		Eigen::Vector3d(150,-150,-125)
	};//*/

	bool result = solvePnP(img_pts, world_pts, 10e-5, 15);
}
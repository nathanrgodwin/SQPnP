/*
BSD 3-Clause License

Copyright (c) 2020, George Terzakis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>
#include <random>

#include <opencv2/core.hpp>

//
// Generate noisy PnP data
static void
GenerateSyntheticPoints(int n,
	cv::Matx<double, 3, 3>& R, //Rotation
	cv::Vec<double, 3>& t, //Translation
	std::vector<cv::Point3_<double>>& points,
	std::vector<cv::Point_<double>>& projections,
	std::vector<cv::Point_<double>>& noisy_projections,
	const double& std_pixel_noise = 0.0,
	const double& radius = 3.0)
{
	assert( n > 2);

	const double std_noise = std_pixel_noise / 1400;
	const double depth = 7*radius; // depth of the barycenter of the points

	const cv::Point3_<double> C(radius / 4, radius / 4, depth );


	// Generate a rotation matrix near the origin
	cv::Vec<double, 3> psi;// = mvnrnd([0; 0; 0], 0.001 * eye(3))';

	static std::default_random_engine generator;
	double sigma_psi = 0.1;
	std::normal_distribution<double> psi_noise(0.0, sigma_psi);
	psi[0] = psi_noise(generator);
	psi[1] = psi_noise(generator);
	psi[2] = psi_noise(generator);

	double sq_norm_psi = psi[0]*psi[0] + psi[1]*psi[1] + psi[2]*psi[2];
	double inv_w = 1.0 / (1 + sq_norm_psi );
	double s = (1 - sq_norm_psi) * inv_w,
	v1 = 2*psi[0] * inv_w,
	v2 = 2*psi[1] * inv_w,
	v3 = 2*psi[2] * inv_w;
	R(0, 0)  = s*s + v1*v1 - v2*v2 - v3*v3;    R(0, 1) = 2*( v1*v2 - s*v3 );              R(0, 2) = 2*( v1*v3 + s*v2 );
	R(1, 0) = 2*( v1*v2 + s*v3 );              R(1, 1) = s*s - v1*v1 + v2*v2 - v3*v3;     R(1, 2) = 2*( v2*v3 - s*v1 );
	R(2, 0) = 2*( v1*v3 - s*v2 );              R(2, 1) = 2*( v2*v3 + s*v1 );              R(2, 2) = s*s - v1*v1 - v2*v2 + v3*v3;

	// Generate a translation that's about 1/25 of the depth
	std::normal_distribution<double> camera_position(0.0, depth/25 );
	cv::Vec<double, 3> pos(camera_position(generator), camera_position(generator), camera_position(generator));

	points.clear();
	projections.clear();
	noisy_projections.clear();
	std::normal_distribution<double> projection_noise(0.0, std_noise );

	while(points.size() < n)
	{
		std::normal_distribution<double> point_X(C.x,  radius);
		std::normal_distribution<double> point_Y(C.y,  radius);
		std::normal_distribution<double> point_Z(C.z,  radius);

		cv::Vec<double, 3> Mw(point_X(generator), point_Y(generator), point_Z(generator)  );
		cv::Vec<double, 3> Mc = R*(Mw - pos);
		if ( Mc[2] < 0 )
		{
			continue;
		}
		cv::Vec<double, 2> proj( Mc[0] / Mc[2], Mc[1] / Mc[2] );
		// Add noise to projection
		cv::Vec<double, 2> noisy_proj = proj;
		noisy_proj[0] += projection_noise(generator);
		noisy_proj[1] += projection_noise(generator);
		noisy_proj[2] += projection_noise(generator);

		points.push_back(Mw);
		projections.push_back(proj);
		noisy_projections.push_back(noisy_proj);
	}

	t = -R*pos;
}
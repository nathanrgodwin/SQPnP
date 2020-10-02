#include <iostream>
#include <chrono>

#include <cv/sqpnp/sqpnp.h>
#include "GenerateSyntheticPoints.h"
#include "ReprojectionError.h"


int main()
{
    int N = 1e3;
    int n = 4;
    double std_pixels = sqrt(7);

    long long avg_duration = 0;
    double avg_error = 0;
    int failures = 0;
    for (int i = 0; i < N; i++)
    {
        cv::Matx<double, 3, 3> Rt;
        cv::Vec<double, 3> tt;
        std::vector<cv::Point3_<double>> points;
        std::vector<cv::Point_<double>> projections;
        std::vector<cv::Point_<double>> noisy_projections;

        GenerateSyntheticPoints(n, Rt, tt, points, projections, noisy_projections, std_pixels);

        std::vector<cv::Mat> rvec, tvec;

        auto start = std::chrono::steady_clock::now();
        cv::sqpnp::PoseSolver opencv_solver;
        opencv_solver.solve(points, noisy_projections, rvec, tvec);

        auto finish = std::chrono::steady_clock::now();
        auto diff = finish - start;

        if (rvec.empty())
        {
            failures++;
            continue;
        }
        avg_duration += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
        avg_error += ReprojectionError(points, projections, rvec[0], tvec[0]);
    }
    avg_duration /= (N - failures);
    avg_error /= (N - failures);

    std::cout << "SQPnP" <<
     "\n Number failures : " << failures <<
     "\n Average execution time : " << avg_duration <<
     "\n Average reprojection RSME : " << avg_error << std::endl;

    return 0;
}
#include <iostream>
#include <chrono>
#include <random>

#define HAVE_OPENCV
#include "sqpnp.h"

#include <opencv2/core/eigen.hpp>

#include <cv/sqpnp/sqpnp.h>

#include "GenerateSyntheticPoints.h"
#include "ReprojectionError.h"


int main()
{
    int N = 1e6;
    double std_pixels = sqrt(7);

    long long cv_avg_duration = 0;
    double cv_avg_error = 0;
    int cv_failures = 0;

    long long eigen_avg_duration = 0;
    double eigen_avg_error = 0;
    int eigen_failures = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(3, 10);

    for (int i = 0; i < N; i++)
    {
        if (i % (int)1e3 == 0)
        {
            std::cout << "\r(" << i << "/" << N << ")";
        }
        int n = distrib(gen);
        cv::Matx<double, 3, 3> Rt;
        cv::Vec<double, 3> tt;
        std::vector<cv::Point3_<double>> points;
        std::vector<cv::Point_<double>> projections;
        std::vector<cv::Point_<double>> noisy_projections;

        GenerateSyntheticPoints(n, Rt, tt, points, projections, noisy_projections, std_pixels);


        std::vector<cv::Mat> rvec1, tvec1;

        cv::sqpnp::PoseSolver opencv_solver;
        bool cv_result = false;
        auto start = std::chrono::steady_clock::now();
        try {
            opencv_solver.solve(points, noisy_projections, rvec1, tvec1);


            auto finish = std::chrono::steady_clock::now();
            auto diff = finish - start;

            if (rvec1.empty())
            {
                cv_failures++;
            }
            else
            {
                cv_avg_duration += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
                cv_avg_error += ReprojectionError(points, projections, rvec1[0], tvec1[0]);
                cv_result = true;
            }
        }
        catch (std::exception& e)
        {
            std::cout << e.what() << std::endl;
            cv_failures++;
        }

        std::vector<cv::Mat> rvec2, tvec2;

        Eigen::Matrix<double, 9, 1> rhat;
        Eigen::Matrix<double, 3, 1> t;
        bool result = false;
        start = std::chrono::steady_clock::now();

        sqpnp::PnPSolver solver(points, noisy_projections);
        if (solver.IsValid())
        {
            result = solver.Solve();
        }

        auto finish = std::chrono::steady_clock::now();
        auto diff = finish - start;

        if (!result || solver.NumberOfSolutions() == 0)
        {
            result = false;
            eigen_failures++;
        }
        else
        {
            rhat = solver.SolutionPtr(0)->r_hat;
            t = solver.SolutionPtr(0)->t;

            finish = std::chrono::steady_clock::now();
            cv::Mat_<double> rhat_cv;
            cv::eigen2cv(rhat, rhat_cv);
            rhat_cv = rhat_cv.reshape(1, 3);
            rvec2.resize(1);
            cv::Rodrigues(rhat_cv, rvec2[0]);
            tvec2.resize(1);
            cv::eigen2cv(t, tvec2[0]);

            eigen_avg_duration += std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
            eigen_avg_error += ReprojectionError(points, projections, rvec2[0], tvec2[0]);
        }
    }
    std::cout << "\r(" << N << "/" << N << ")" << std::endl;
    if (cv_failures != N)
    {
        cv_avg_duration /= (N - cv_failures);
        cv_avg_error /= (N - cv_failures);
    }
    else
    {
        cv_avg_error = std::nan("");
        cv_avg_duration = std::nan("");
    }

    if (eigen_failures != N)
    {
        eigen_avg_duration /= (N - eigen_failures);
        eigen_avg_error /= (N - eigen_failures);
    }
    else
    {
        eigen_avg_duration = std::nan("");
        eigen_avg_error = std::nan("");
    }

    std::cout << "OpenCV SQPnP" <<
     "\n Number failures : " << cv_failures <<
     "\n Average execution time : " << cv_avg_duration <<
     "\n Average reprojection RSME : " << cv_avg_error << std::endl;

    std::cout << "Eigen SQPnP" <<
        "\n Number failures : " << eigen_failures <<
        "\n Average execution time : " << eigen_avg_duration <<
        "\n Average reprojection RSME : " << eigen_avg_error << std::endl;

    return 0;
}
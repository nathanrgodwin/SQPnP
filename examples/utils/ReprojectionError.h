#pragma once

#include <opencv2/calib3d.hpp>

static double
ReprojectionError(const std::vector<cv::Point3_<double>>& points,
    const std::vector<cv::Point_<double>>& projections,
    const cv::Mat& rvec,
    const cv::Mat& tvec)
{
    cv::Matx<double, 3, 3> K(1, 0, 0,
        0, 1, 0,
        0, 0, 1);
    std::vector<cv::Point_<double>> est;
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    cv::projectPoints(points, rvec, tvec, K, dist_coeffs, est);
    cv::Mat imagePoints(projections, false);
    cv::Mat projectedPoints(est, false);
    double rmse = cv::norm(projectedPoints, imagePoints, cv::NORM_L2) / sqrt(2 * est.size());
    return rmse;
}
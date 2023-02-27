#pragma once

#include "helper.hpp"



///
/// search algo. of motion compensation
///
cv::Mat motion_estimation_full_search( const cv::Mat &refFrame, const cv::Mat &curFrame, int blockSize, int searchRange);

cv::Mat motion_estimation_2dlog_search(const cv::Mat &refFrame, const cv::Mat &curFrame, int blockSize, int searchRange);


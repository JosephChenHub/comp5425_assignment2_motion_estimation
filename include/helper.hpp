#pragma once

#include <opencv2/opencv.hpp>

#define ASSERT_FRAMES(refFrame, curFrame) \
do { \ 
    CV_Assert(refFrame.size() == curFrame.size()); \
    CV_Assert(refFrame.type() == curFrame.type()); \
} while(false)


cv::Mat predict_frame(const cv::Mat& refFrame, const cv::Mat &motionVectors, const int blockSize);
float compute_mse(const cv::Mat& original, const cv::Mat& predicted);


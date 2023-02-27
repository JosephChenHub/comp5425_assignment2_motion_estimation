#include "helper.hpp"

cv::Mat predict_frame(const cv::Mat& refFrame, const cv::Mat &motionVectors, const int blockSize) {
    // Initialize output frame with same size and type as reference frame
    cv::Mat curFrame = cv::Mat::zeros(refFrame.size(), refFrame.type());

    // Loop through all blocks in the image
    for (int y = 0; y < curFrame.rows; y += blockSize) {
        for (int x = 0; x < curFrame.cols; x += blockSize) {
            // Get the motion vector for this block
            cv::Point2f motion = motionVectors.at<cv::Point2f>(y / blockSize, x / blockSize);

            // Calculate the coordinates of the top-left corner of the block in the reference frame
            int refX = x + motion.x; // here we directly convert float to int 
            int refY = y + motion.y;

            // Make sure the block is within the bounds of the reference frame
            if (refX >= 0 && refX + blockSize <= refFrame.cols &&
                refY >= 0 && refY + blockSize <= refFrame.rows) {
                // Copy the block from the reference frame to the current frame at the new location
                cv::Rect roi(refX, refY, blockSize, blockSize);
                cv::Mat block = refFrame(roi);
                block.copyTo(curFrame(cv::Rect(x, y, blockSize, blockSize)));
            }
        }
    }

    return curFrame;
}

float compute_mse(const cv::Mat& original, const cv::Mat& predicted) {
    cv::Mat err;
    cv::absdiff(original, predicted, err);
    return (float)mean(err.mul(err))[0];
}


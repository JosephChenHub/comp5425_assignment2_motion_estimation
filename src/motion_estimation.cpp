#include "motion_estimation.hpp"
#include <vector>
#include <limits>

using namespace cv;
using namespace std;



Mat motion_estimation_full_search(const Mat &refFrame, const Mat &curFrame, int blockSize, int searchRange) {
    ASSERT_FRAMES(refFrame, curFrame);
    int rows = refFrame.rows;
    int cols = refFrame.cols;
    int mv_rows = rows / blockSize;
    int mv_cols = cols / blockSize;
    Mat motionVectors(mv_rows, mv_cols, CV_32FC2); // we use float to store motion vectors
    
    // Loop over every block in current frame
    for (int y = 0; y < rows; y += blockSize) {
        for (int x = 0; x < cols; x += blockSize) {
            float minMAD = std::numeric_limits<float>::max();
            Point2f bestMV(0, 0);
            
            // time complexity: (2p+1)**2 * N**2 * 3, N: block numbers, p: searchRange
            for (int m = -searchRange; m <= searchRange; m++) {
                for (int n = -searchRange; n <= searchRange; n++) {
                    if (y+m < 0 || y+m+blockSize > rows||
                        x+n < 0 || x+n+blockSize > cols ) {
                        continue;
                    }
                    // Compute MAD between the current block and the reference block
                    float MAD = 0;
                    for(int i = 0; i < blockSize; ++i) {
                       for(int j = 0; j < blockSize; ++j) {
                           float refPixel = refFrame.at<uchar>(y+m+i, x+n+j);
                           float curPixel = curFrame.at<uchar>(y+i, x+j);
                           MAD += fabs(refPixel - curPixel);
                       }
                    }
                    MAD /= (blockSize * blockSize);
                    
                    // Update the motion vector if the current MAD is smaller than the reference minimum
                    if (MAD < minMAD) {
                        minMAD = MAD;
                        bestMV = Point2f(n, m);
                    }
                }
            }
            
            // Store the best motion vector for the current block
            motionVectors.at<Point2f>(y / blockSize,  x / blockSize) = bestMV;
        }
    }
    
    return motionVectors;
}


Mat motion_estimation_2dlog_search(const Mat &refFrame, const Mat &curFrame, int blockSize, int searchRange) {
    ASSERT_FRAMES(refFrame, curFrame);

    // Initialize variables
    int rows = refFrame.rows;
    int cols = refFrame.cols;
    int mv_rows = rows / blockSize;
    int mv_cols = cols / blockSize;
    Mat motionVectors(mv_rows, mv_cols, CV_32FC2); // we use float to store motion vectors

    // Loop over every block
    for (int y = 0; y < rows; y += blockSize) {
        for (int x = 0; x < cols; x += blockSize) {   
            float minMAD = std::numeric_limits<float>::max();
            Point2f bestMV(0, 0);
            Point2f center(x, y); // start point

            for(int log_scale = log2(searchRange); log_scale >= 0; log_scale--) {
                int step_size = 1 << log_scale; // step size will be reduced gradually
                
                // loop over search points (8+1 candidate points) 
                // to find a optimal candidate, e.g., [center[0] - step_size, center[1]] is a left point to the center point
                // with a distance step_size, [center[0]+step_size, center[1]] is a right point  to the center.
                // hence, there exist 8 neighours should be compared to the center.

                 throw std::runtime_error("TODO: please implement 2d log. search");    
                // update center with optimal candidate
            }
            motionVectors.at<Point2f>(y / blockSize,  x / blockSize) = bestMV;
        }
    }
    return motionVectors;
}


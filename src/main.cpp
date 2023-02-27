#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <chrono>

using namespace std::chrono;

#include "motion_estimation.hpp"





struct Arguments {
    std::string refFrameName;
    std::string curFrameName;
    int blockSize;
    int searchRange;
};

Arguments parseArguments(int argc, const char** argv) {
    Arguments args;

    // Map of option names to their values
    std::unordered_map<std::string, std::string> options;

    // Parse the arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.substr(0, 1) == "-") {
            // Found an option
            std::string optName = arg.substr(1);
            i++;
            if (i >= argc) {
                // Error: option is missing a value
                throw std::runtime_error("Option " + optName + " requires a value.");
            }
            std::string optValue = argv[i];
            options[optName] = optValue;
        } else {
            // Found an argument (not an option)
            // This is an error if we've already found an input file name
            if (args.refFrameName != "") {
                throw std::runtime_error("Multiple input file names specified.");
            }
            args.refFrameName = arg;
        }
    }
    args.searchRange = 15;
    args.blockSize = 16;

    // Validate the options and their values
    if (options.count("r")) {
        args.refFrameName  = options["r"];
    } else {
        throw std::runtime_error("Reference frame name not specified. Usage example: ./motion_estimation -r xxx.jpg -c xxx.jpg -b 16 -s 15");
    } 
    if (options.count("c")) {
        args.curFrameName = options["c"];
    } else {
        throw std::runtime_error("Current frame name not specified. Usage example: ./motion_estimation -r xxx.jpg -c xxx.jpg -b 16 -s 15");
    }
    if (options.count("b")) {
        args.blockSize = std::stoi(options["b"]);
    }
    if (options.count("s")) {
        args.searchRange = std::stoi(options["s"]);
    }
    return args;
}



int main(int argc, const char** argv) {
    try {
        Arguments args = parseArguments(argc, argv);
        std::cout << "Reference frame: " << args.refFrameName << std::endl;
        std::cout << "Current frame: " << args.curFrameName << std::endl;

        cv::Mat refFrame = cv::imread(args.refFrameName, cv::IMREAD_GRAYSCALE);
        cv::Mat curFrame = cv::imread(args.curFrameName, cv::IMREAD_GRAYSCALE); 
        ASSERT_FRAMES(refFrame, curFrame);
        int blockSize = args.blockSize;
        int searchRange = args.searchRange; 
        /// pad image with zeros
        int rowsToPad = (blockSize - refFrame.rows % blockSize) % blockSize;
        int colsToPad = (blockSize - refFrame.cols % blockSize) % blockSize;
        if (rowsToPad != 0 || colsToPad != 0) {
            copyMakeBorder(refFrame, refFrame, 0, rowsToPad, 0, colsToPad, cv::BORDER_CONSTANT, cv::Scalar(0));
            copyMakeBorder(curFrame, curFrame, 0, rowsToPad, 0, colsToPad, cv::BORDER_CONSTANT, cv::Scalar(0));
        }        
        
        auto start1 = high_resolution_clock::now(); 
        cv::Mat motionVectors1 = motion_estimation_full_search(refFrame, curFrame, blockSize, searchRange);
        auto end1 = high_resolution_clock::now();
        float cost1 = duration_cast<milliseconds>(end1 - start1).count();

        auto start2 = high_resolution_clock::now(); 
        cv::Mat motionVectors2 = motion_estimation_2dlog_search(refFrame, curFrame, blockSize, searchRange);
        auto end2 = high_resolution_clock::now();
        float cost2 = duration_cast<milliseconds>(end2 - start2).count();


        cv::Mat predictFrame1 = predict_frame(refFrame, motionVectors1, blockSize);
        cv::Mat predictFrame2 = predict_frame(refFrame, motionVectors2, blockSize);

        std::cout << " Frame shape:" << curFrame.cols << " * " << curFrame.rows << std::endl;
        std::cout << " block size:" << blockSize  << " * " << blockSize
                  << " search range: [" << -searchRange  << "," 
                  << searchRange << "]" << std::endl;
        std::cout << " MSE of full search:" << compute_mse(curFrame, predictFrame1)
                  << " cost time:" << cost1 << " ms" << std::endl;
        std::cout << " MSE of 2d log search:" << compute_mse(curFrame, predictFrame2) 
                  << " cost time:" << cost2 << " ms" << std::endl;

        std::cout << " visualization:" << std::endl;        
        // Visualize motion vectors
        cv::Mat motionVectorImg = cv::Mat::zeros(refFrame.size(), CV_8UC3);
        for (int i = 0; i < refFrame.rows; i += blockSize) {
            for (int j = 0; j < refFrame.cols; j += blockSize) {
                cv::Point2f mv = motionVectors2.at<cv::Point2f>(i / blockSize, j / blockSize);
                int arrowLength = mv.x * mv.x + mv.y * mv.y;
                arrowLength = std::sqrt(arrowLength);
                cv::arrowedLine(motionVectorImg, cv::Point(j, i), cv::Point(j + mv.y, i + mv.x), cv::Scalar(0, 255, 0), 1);
            }
        }
        // Display input frames and motion vectors
        cv::imshow("Reference Frame", refFrame);
        cv::imshow("Current Frame", curFrame);
        cv::imshow("Motion Vectors", motionVectorImg);
        cv::imshow("Predict Frame", predictFrame2);
        cv::waitKey(0);
        cv::imwrite("./data/predict.jpg", predictFrame2);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

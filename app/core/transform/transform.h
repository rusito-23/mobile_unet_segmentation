#ifndef TRANSFORM_H
#define TRANSFORM_H

// includes
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#pragma clang pop

// class definition

class Transformer {
public:
  cv::Mat blur_background(cv::Mat image, cv::Mat mask);
  cv::Mat replace_background(cv::Mat image, cv::Mat mask, cv::Mat background);
private:
  CvSize size;
};

#endif

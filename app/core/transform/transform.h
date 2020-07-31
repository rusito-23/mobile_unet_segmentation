#ifndef TRANSFORM_H
#define TRANSFORM_H

// includes

#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

// class definition

class Transformer {
public:
  cv::Mat blur_background(cv::Mat image, cv::Mat mask);
  cv::Mat replace_background(cv::Mat image, cv::Mat mask, cv::Mat background);
  Transformer(CvSize, bool, bool);
private:
  CvSize size;
  bool show_raw_mask;
  bool show_blurred_mask;
};

#endif

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

/**
 Transformer
 
 Handles different transformations that can be made using
 the original image and the given mask.
 
 @method blur_background Blurs image where the mask pixels are black
 @method replace_background Replaces image with given background where the mask pixels are black
 */
class Transformer {
public:
    
    /**
     Blur Background
     
     Blurs the given image using the corresponding mask, considering the mask given
     was computed using the original image.
     
     @param image OpenCV Mat containing Original image - BGR - 8CU3
     @param mask OpenCV Mat resulting from passing the original image through a segmentation model.
                 BGR - 8CU3 (must contain 0's and 255's)
     @return OpenCV Mat containing the resulting image - BGR - 8CU3
     */
    cv::Mat blur_background(cv::Mat image, cv::Mat mask);
    
    /**
     Replace background
     
     Replaces pixels in original image, corresponding with black pixels in the mask with the given background.
     
     @param image OpenCV Mat containing Original image - BGR - 8CU3
     @param mask OpenCV Mat resulting from passing the original image through a segmentation model.
                 BGR - 8CU3 (must contain 0's and 255's)
     @param background OpenCV Mat containing background to perform replacement. BGR - 8CU3
     */
    cv::Mat replace_background(cv::Mat image, cv::Mat mask, cv::Mat background);

private:
    CvSize size;
};

#endif

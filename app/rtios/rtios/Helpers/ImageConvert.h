//
//  UIImage+SampleBuffer.h
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

#ifdef __cplusplus
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import <opencv2/core/core.hpp>
#pragma clang pop
#endif

NS_ASSUME_NONNULL_BEGIN

@interface ImageConvert: NSObject
#ifdef __cplusplus

+ (cv::Mat) cvMatFromSampleBuffer:(CMSampleBufferRef) sampleBuffer;
+ (UIImage *) uiImageFromCvMat:(cv::Mat) mat;
+ (UIImage *) uiImageFromGrayCvMat:(cv::Mat)mat;
+ (cv::Mat) cvMatFromUIImage:(UIImage *)image;

#endif
@end

NS_ASSUME_NONNULL_END

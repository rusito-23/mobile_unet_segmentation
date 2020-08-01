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

@interface UIImage (Conversions)
#ifdef __cplusplus

+ (cv::Mat) matFromSampleBuffer:(CMSampleBufferRef) sampleBuffer;
+ (UIImage *) imageFromMat:(cv::Mat) mat;

#endif
@end

NS_ASSUME_NONNULL_END

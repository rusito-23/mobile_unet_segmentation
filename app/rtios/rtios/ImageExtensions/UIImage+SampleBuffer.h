//
//  UIImage+SampleBuffer.h
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface UIImage (SampleBuffer)

+ (id)fromSampleBuffer:(CMSampleBufferRef) sampleBuffer;

@end

NS_ASSUME_NONNULL_END

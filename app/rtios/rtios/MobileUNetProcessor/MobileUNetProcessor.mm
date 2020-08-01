//
//  MobileUNetProcessor.mm
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "MobileUNetProcessor.h"
#import "UIImage+Conversions.h"
#import <opencv2/core/core.hpp>

@interface MobileUNetProcessor ()
@property (weak, nonatomic) NSObject<MobileUnetProcessorDelegate> *delegate;
@end

@implementation MobileUNetProcessor

- (id) initWithDelegate:(NSObject<MobileUnetProcessorDelegate> *) delegate {
    if (self = [super init]) {
        self.delegate = delegate;
    }
    return self;
}

- (void) captureSession:(CaptureSession *) captureSession didCaptureFrame:(CMSampleBufferRef) frame {
    cv::Mat matFrame = [UIImage matFromSampleBuffer:frame];
    UIImage *imageFrame = [UIImage imageFromMat:matFrame];
    [self.delegate processor:self didProcessFrame:imageFrame];
}

@end

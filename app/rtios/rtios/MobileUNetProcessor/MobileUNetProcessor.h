//
//  MobileUNetProcessor.h
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "CaptureSession.h"

NS_ASSUME_NONNULL_BEGIN

@class MobileUNetProcessor;

@protocol MobileUnetProcessorDelegate
- (void) processor:(MobileUNetProcessor *) processor didProcessFrame:(UIImage *)frame;
@end

@interface MobileUNetProcessor : NSObject <CaptureSessionDelegate>

- (id) initWithDelegate:(NSObject<MobileUnetProcessorDelegate> *)delegate;
- (void) captureSession:(CaptureSession *)captureSession didCaptureFrame:(CMSampleBufferRef)frame;

@end

NS_ASSUME_NONNULL_END

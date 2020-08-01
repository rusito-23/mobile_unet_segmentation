//
//  ViewController.m
//  rtios
//
//  Created by Igor on 31/07/2020.
//  Copyright © 2020 rusito23. All rights reserved.
//

#import "ViewController.h"
#import "CaptureSession.h"
#import "MobileUNetProcessor.h"

@interface ViewController () <MobileUnetProcessorDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) CaptureSession *captureSession;
@property (strong, nonatomic) MobileUNetProcessor *processor;
@end

@implementation ViewController

#pragma mark - Lifecycle

- (void) viewDidLoad {
    [super viewDidLoad];
    
    // setup views
    self.imageView.contentMode = UIViewContentModeScaleAspectFill;
    
    // setup capture session
    self.processor = [[MobileUNetProcessor alloc] initWithDelegate:self];
    self.captureSession = [[CaptureSession alloc] initWithDelegate:self.processor];
}

- (void) viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self.captureSession startSession];
}

- (void) viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    [self.captureSession endSession];
}

#pragma mark - Processor Delegate

- (void) processor:(MobileUNetProcessor *) processor didProcessFrame:(UIImage *)frame {
    self.imageView.image = frame;
}

#pragma mark - Prevent rotation

- (BOOL) shouldAutorotate {
    return NO;
}

- (UIInterfaceOrientation) preferredInterfaceOrientationForPresentation {
    return UIInterfaceOrientationPortrait;
}

@end


# Mobile Unet Segmentation

Portrait segmentation in Keras, using [divamgupta's segmentation models](https://github.com/divamgupta/image-segmentation-keras) source code. Aiming to perform real time segmentation in mobile and desktop devices, using a common source code written in C++ with Tensorflow Lite.

## Demo

| RTCPP (MBP 2017 - 2,9 GHz Quad-Core Intel Core i7) | RTIOS (iPhone 8) |
| :---: | :---: |
| <img alt="demo_rtcpp" src="./demo/demo_rtcpp.gif" height=400/> | <img alt="demo_gif" src="./demo/demo_rtios.gif" height=400/> |

## Content

- [Train](#train)
- [Real Time Applications](#real-time-applications)
    - [Core](#core)
    - [RTCPP](#rtcpp)
    - [RTPY](#rtpy)
    - [RTIOS](#rtios)
- [Future Work](#todo)


## Train

This model was built/trained using Keras, the model was refactored from [divamgupta's segmentation models](https://github.com/divamgupta/image-segmentation-keras) source code.

### Dataset

The dataset used is a frankestein made from the [supervisely dataset](http://supervise.ly/). The final dataset can be found [here](https://drive.google.com/drive/folders/1uVEgfBRE2x_RnPYRUj0QL6AJNPfvxYHM?usp=sharing).

### Pipeline Features

- data augmentation with [albumentations](https://github.com/albumentations-team/albumentations)
- [YACS](https://github.com/rbgirshick/yacs) to manage experiment configs (these can be found in [here](./configs))
- Telegram callback notifications 
- Creates an output folder and stores:

    - Tensorboard logs
    - Logger output
    - Model Checkpoints

- SLURM [scripts](./slurm) to enqueue the training process in a computer with shared resources.

### Model

The model consists of a MobileNet backbone and a UNet head. The MobileNet is prepared using fchollet's pretrained weights.
The full-model weights, along with corresponding output information, can be found in [here](https://drive.google.com/drive/folders/1fvmbBIBeCga2cKpGz47mRe98OCkQ_TOF?usp=sharing).
This folder contains the pretrained weights for keras and tflite to test the realtime applications. These need to be located in `models/model.h5` or `models/model.tflite` to be used by the applications.

## Real Time Applications

As the idea is to mantain a model that can run in real-time in several devices, along with the model training, there is a C++ Application to run this.

### Tensorflow Lite Compile

The Core and all depending applications need Tensorflow Lite library to run. In order to achieve this, Tensorflow Lite must be compiled from source, therefore, the tensorflow dependency is set as a submodule (v1.13x). To compile, use:

- `./tensorflow/lite/tools/make/download_dependencies.sh`
- `./tensorflow/lite/tools/make/build_ios_universal_lib.sh` to build for iOS
- `./tensorflow/lite/tools/make/build_lib.sh` to build for the current arch (tried on macOS)

### Core

C++ Shareable Source Code to Perform the inference using the TFLite model and other transformations in the image (blur/replace the background).
Refer to [the source code](./app/core) for more info.

### RTCPP

C++ Application to test the Core. Uses OpenCV to load frames from Camera input and passes them through the model.
This app was made using macOS, therefore the Makefile link Tensorflow Lite library agains the path `tensorflow/lite/tools/make/gen/osx_x86_64/lib`, which is generated using the script `tensorflow/lite/tools/make/build_lib.sh`, if you ran this script using other SO, please modify the Makefile to link against the correct lib.


### RTPY

Python application to test the performance of the raw Keras model (without TFLite conversion).


### RTIOS

iOS application that uses the [Core](#core) to perform background segmentation in iOS devices. This application was set to work with `arch64` only.
It uses OpenCV, using the Cocoapods dependency manager. To set up, run `pod install` in `rtios` root folder. The `model.tflite` file needs to be copied into `rtios/rtios` folder as well. 

### TODO

- [ ] CORE error handling 
- [ ] Improve FPS:
    - [X] Downscale to ~512/224 to perform the blur
    - [ ] Perform mask prediction and blur in different threads
- [ ] Perform the thread handling in C++ (if possible)
- [ ] Generate a new synthetic dataset (maybe MaskRCNN) to get better background replacement results (generate dataset using selfies and videoconference scenarios)

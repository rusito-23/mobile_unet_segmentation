
# Mobile Unet Segmentation

Portrait segmentation in Keras, using [divamgupta's segmentation models](https://github.com/divamgupta/image-segmentation-keras) source code. Aiming to perform real time segmentation in mobile and desktop devices, using a common source code written in C++ with Tensorflow Lite.

## Demo

| RTCPP (MBP 2017 - 2,9 GHz Quad-Core Intel Core i7) |
| :---: |
| ![rtcpp demo](./demo/demo_rtcpp.gif) |

| RTIOS (iPhone 8) |
| :---: |
| ![rtios demo](./demo/demo_rtios.gif) |

## Content

- [Train](#train)
- [Real Time Applications](#real-time-applications)
    - [Core](#core)
    - [RTCPP](#rtcpp)
    - [RTPY](#rtpy)
    - [RTIOS](#rtios)


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

## Real Time Applications

As the idea is to mantain a model that can run in real-time in several devices, along with the model training, there is a C++ Application to run this.

### Tensorflow Lite Compile

The Core and all depending applications need Tensorflow Lite library to run (used tag v1.13x). In order to achieve this, Tensorflow Lite must be compiled from source, here are the steps:

- Clone the tensorflow repository, replacing the [tensorflow folder](./app/tensorflow).
```
rm -rf app/tensorflow
git clone https://github.com/tensorflow/tensorflow.git app/tensorflow
```
- Run the corresponding scripts to generate the neccesary libraries, located in `./tensorflow/lite/tools/make`. I used `build_lib.sh` and `build_ios_universal_lib.sh`.


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

### TODO's

- error handling in the Core (use custom exceptions?)
- background replacement for iOS app
- precict the mask in background, but blur in main thread
- maybe perform the thread handling in C++

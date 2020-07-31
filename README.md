
# Mobile Unet Segmentation

Portrait segmentation in Keras, using [divamgupta's segmentation models](https://github.com/divamgupta/image-segmentation-keras) source code. Aiming to perform real time segmentation in mobile and desktop devices, using a common source code written in C++ with Tensorflow Lite.

## Contents

- [Train](./train)

    Model source code and training pipeline. Uses data augmentation with [albumentations](https://github.com/albumentations-team/albumentations) and [YACS](https://github.com/rbgirshick/yacs) to manage experiment configs, as well as other little features such as Telegram notifications, logger, etc...

- [Slurm](./slurm)

    This folder contains some scripts to run the training pipeline within a slurm job and is used to store the different experiment logs.

- [Configs](./configs)

    Experiment configurations written in YAML for YACS.

- [App](./app)

    - [Core](./app/core)

        C++ Shareable Source Code to Perform the inference using the TFLite model and other transformations in the image (blur/replace the background).

    - [RTCPP](./app/rtcpp)

        C++ Application to test the Core. Uses OpenCV to load frames from Camera input and passes them through the model.

    - [RTPY](./app/rtpy)
		
		Python application to test the performance of the raw Keras model (without TFLite conversion).
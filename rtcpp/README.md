# Real Time Test with ONNX

This folder contains a C++ application which allows us to perform Real Time inference using the ONNX converted model over [the onnxruntime CPP API](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_cxx_api.h).

## Build

The [Makefile](Makefile) uses **pkg-config** to setup the required libraries path.

The requirements are:

- [opencv](https://opencv.org)
- [onnxruntime](https://microsoft.github.io/onnxruntime/)
- [libzip](https://libzip.org)

In macOS, these requirements can de downloaded using:

```
$ brew install opencv onnxruntime libzip
```

Unfortunately, *onnxruntime* does not provides a default *.pc* file to setup the libraries, this file can be found in [this folder](pkg-conf). It's already exported into the pkg conf path inside the makefile.

## Usage

The executable will be generated in `./realtime_test`.

Parameters available are:

- **model-file** (required): path to the model.onnx file
- **show-raw-mask** (default: false): flag to show the raw mask
- **show-blurred-mask** (default: false): flash to show the blurred mask
- **width** (default: 600): preview width
- **height** (default: 400): preview height
- **background** (default: NO): if set to a valid image, uses this image to replace the background
- **threshold** (default: 0.5): segmentation threshold

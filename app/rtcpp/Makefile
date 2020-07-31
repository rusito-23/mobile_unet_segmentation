
CC = g++
CFLAGS = -Werror -Wall
STD = c++11

TARGET = realtime_test
SOURCE = main.cpp ../core/transform/transform.cpp ../core/predict/predict.cpp

PKG_LIBS = opencv4 libzip
PKG_FLAGS = `pkg-config --cflags --libs ${PKG_LIBS}`

TF_INCLUDE = ../tensorflow
TF_LIB = ../tensorflow/tensorflow/lite/tools/make/gen/osx_x86_64/lib
TFLITE_LIB = tensorflow-lite

CORE_INCLUDE = ../core

all:
	${CC} ${CFLAGS} -std=${STD} ${SOURCE} \
	    ${PKG_FLAGS} \
	    -I${TF_INCLUDE} -I${CORE_INCLUDE} \
	    -L${TF_LIB} -l${TFLITE_LIB} \
	    -o ${TARGET}  

clean:
	rm ${TARGET}

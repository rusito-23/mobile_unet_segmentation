
# COMPILER + FLAGS + STD
CC = g++
CFLAGS = -Werror -Wall -Wno-unknown-pragmas
STD = c++11

# SOURCE OPTS
TARGET = realtime_test
SRC = main.cpp

# CORE OPTS
CORE_SRC = ../core/transform/transform.cpp ../core/predict/predict.cpp
CORE_INCLUDE = ../core
CORE_FLAGS = -I${CORE_INCLUDE} ${CORE_SRC}

# EXTERNAL DEPENDENCIES
PKG_LIBS = opencv4 libzip
PKG_FLAGS = `pkg-config --cflags --libs ${PKG_LIBS}`

# TFLITE OPTS
TFLITE_INCLUDE = ../tensorflow
TFLITE_LIB = ../tensorflow/tensorflow/lite/tools/make/gen/osx_x86_64/lib
TFLITE = tensorflow-lite
TFLITE_FLAGS = -I${TFLITE_INCLUDE} -L${TFLITE_LIB} -l${TFLITE}


all:
	${CC} ${CFLAGS} -std=${STD} ${SRC} -o ${TARGET} \
	    ${PKG_FLAGS} \
	    ${CORE_FLAGS} \
	    ${TFLITE_FLAGS}

clean:
	rm ${TARGET}

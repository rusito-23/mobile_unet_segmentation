
CC = g++
CFLAG = -Werror -Wall
TARGET = realtime_test
SOURCE = main.cpp source/transform/transform.cpp source/predict/predict.cpp
STD = c++14
LIBS = opencv4 libzip onnxruntime
PWD = $(shell pwd)
PKG_FLAGS = $(shell export PKG_CONFIG_PATH=${PWD}/pkg-conf/ ;\
	    pkg-config --cflags --libs ${LIBS})

all:
	@${CC} ${CFLAGS} ${PKG_FLAGS} -o ${TARGET} ${SOURCE} -std=${STD}

clean:
	rm ${TARGET}

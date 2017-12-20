# compiler
CC = gcc -O3 -g -Wall 
CXX = g++ -O3 -g -Wall 
LINK = g++ -O3 -g -Wall

#CC = icc -O3 -g -Wall -std=c99 -openmp -m64 -xHOST -ip -ansi-alias -fno-alias -vec-report3
#CXX = icpc -O3 -g -Wall -std=c++0x -Wno-deprecated -openmp -m64 -xHOST -ip -ansi-alias -fno-alias -vec-report3

CFLAGS = -std=c99 -fopenmp -m64 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -m64
CXXFLAGS = -std=c++0x -Wno-deprecated -fopenmp -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -m64
#CFLAGS = -std=c99 -openmp -m64 -xHOST -ip -ansi-alias -fno-alias -vec-report3
#CXXFLAGS = -std=c++0x -Wno-deprecated -openmp -m64 -xHOST -ip -ansi-alias -fno-alias -vec-report3
CFLAGS += -D_SSE2
CXXFLAGS += -DHAVE_MKL
LINK += -fopenmp
CXXFLAGS += -DDEBUG

NVCC = nvcc -ccbin gcc

CUDA_FLAGS = -Xcompiler -fopenmp --ptxas-options=-v
GENCODE_FLAGS = -m64 -gencode arch=compute_50,code=sm_50
#GENCODE_FLAGS = -m64 -gencode arch=compute_20,code=sm_20
CUDA_FLAGS += $(GENCODE_FLAGS)

# include
# cuda
CUDA_PATH = /home/xiaocen/Software/cuda/cuda-8.0
#CUDA_PATH = /home/zx/Software/cuda-7.5
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_COMMON_INCLUDE = /home/xiaocen/Software/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc
#CUDA_COMMON_INCLUDE = $(CUDA_PATH)/samples/common/inc

# opencv
OPENCV_PATH = /home/xiaocen/Software/opencv
OPENCV_INCLUDE = $(OPENCV_PATH)/include
#OPENCV_PATH = /home/zx/Software/opencv
#OPENCV_INCLUDE = $(OPENCV_PATH)/include

# lapack mkl
LAPACKE_PATH = /home/xiaocen/Software/lapack/lapack_build
LAPACKE_INCLUDE = $(LAPACKE_PATH)/include
#INTEL_INCLUDE = /home/zx/intel/include
#MKL_PATH = /home/zx/intel/composer_xe_2013_sp1/mkl
#MKL_INCLUDE = $(MKL_PATH)/include
#MKL_FFTW_INCLUDE = $(MKL_PATH)/include/fftw

INCLUDES = -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE) -I$(LAPACKE_INCLUDE)
#INCLUDES = -I$(INTEL_INCLUDE) -I$(MKL_INCLUDE) -I$(MKL_FFTW_INCLUDE) -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)

# library
LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64 -L$(LAPACKE_PATH)/lib
#LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64 -L$(MKL_PATH)/lib/intel64

LDFLAGS_OPENCV = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
LDFLAGS_LAPACK = -llapacke -llapack -lcblas -lblas -lgfortran
LDFLAGS = -lm -lpthread -lcudart -lcublas
LDFLAGS += $(LDFLAGS_LAPACK)
#LDFLAGS += $(LDFLAGS_OPENCV)

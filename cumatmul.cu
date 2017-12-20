#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
using std::cout;

#define GNU_C_COMPILER
#if defined(GNU_C_COMPILER)
extern "C" {
#include "cblas.h"
#include "lapacke.h"
#include "lapacke_mangling.h"
}
#elif defined(INTEL_C_COMPILER)
#include "mkl.h"
#endif


#include <cuda_runtime.h>
#include <cublas_v2.h>

#define EXIT_FAILURE 1

#define nullptr NULL

#define USE_PAGE_LOCK_MEMORY
//#define USE_MALLOC_PITCH

template<typename T>
T * host_flatten_malloc_aligned(const long int n1, const int Aligned)
{
	long int nbts = sizeof(T) * n1;
	if(n1 < 0) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
		return nullptr;
	}
//	T * ptr __attribute__((aligned(Aligned))) = (T*)_mm_malloc(nbts, Aligned);
	T * ptr = (T*)malloc(nbts);
	return ptr; 
}

template<typename T>
void host_flatten_free_aligned(T * ptr)
{
//	if(ptr != nullptr) _mm_free(ptr);
	if(ptr != nullptr) free(ptr);
	ptr = nullptr;
}

void initData(float * const ptr, const int M, const int N, bool inverse)
{
	for(int i = 0; i < M; i++) {
		float * row_ptr = &ptr[i];
		for(int j = 0; j < N; j++) {
//			*row_ptr = inverse ? 1.f / (i + j + 2) : i + j + 2;
			*row_ptr = 1.0;
			row_ptr += M;
		}
	}
}

int main(int argc, char * argv[])
{
	if(argc != 4) {
		fprintf(stderr, "USAGE: M N K\n");
		exit(EXIT_FAILURE);
	}

	const int M = atoi(argv[1]);
	const int N = atoi(argv[2]);
	const int K = atoi(argv[3]);
	const int Aligned = 32;	

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	long int entries_a = (long)M * N;
	long int entries_b = (long)N * K;
	long int entries_c = (long)M * K;
	long int nbts_a = sizeof(float) * (long)M * N;
	long int nbts_b = sizeof(float) * (long)N * K;
	long int nbts_c = sizeof(float) * (long)M * K;
#if defined(USE_PAGE_LOCK_MEMORY)
	float * h_a;
	cudaStat = cudaMallocHost((void**)&h_a, nbts_a);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes from pinned host memory, file: %s, line: %d\n", nbts_a, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	float * h_b;
	cudaStat = cudaMallocHost((void**)&h_b, nbts_b);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes from pinned host memory, file: %s, line: %d\n", nbts_b, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	float * h_c;
	cudaStat = cudaMallocHost((void**)&h_c, nbts_c);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes from pinned host memory, file: %s, line: %d\n", nbts_c, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
#else
	float * h_a = host_flatten_malloc_aligned<float>(entries_a, Aligned);
	assert(h_a != nullptr);

	float * h_b = host_flatten_malloc_aligned<float>(entries_b, Aligned);
	assert(h_b != nullptr);

	float * h_c = host_flatten_malloc_aligned<float>(entries_c, Aligned);
	assert(h_c != nullptr);
#endif	

	initData(h_a, M, N, true);
	initData(h_b, N, K, false);
	memset(h_c, 0, nbts_c);
/*
	cout << "Matrix A: \n";
	for(int i = 0; i < M; i++) {
		float * a_ptr = &h_a[i];
		for(int j = 0; j < N; j++) {
			if(j < N - 1) {
				fprintf(stdout, "%.7e\t", *a_ptr);
			} else {
				fprintf(stdout, "%.7e\n", *a_ptr);
			}
			a_ptr += M;
		}
	}

	cout << "Matrix B: \n";
	for(int i = 0; i < N; i++) {
		float * b_ptr = &h_b[i];
		for(int j = 0; j < K; j++) {
			if(j < K - 1) {
				fprintf(stdout, "%.7e\t", *b_ptr);
			} else {
				fprintf(stdout, "%.7e\n", *b_ptr);
			}
			b_ptr += N;
		}
	}
*/
	float * d_a;
	float * d_b;
	float * d_c;
#if defined(USE_MALLOC_PITCH)
	int pitch_a = 0;
	cudaStat = cudaMallocPitch((void**)&d_a, &pitch_a, sizeof(float) * M, N);	
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %dx%d 2d matrix using routine cudaMallocPitch(...), file: %s, line: %d\n", M, N, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	int pitch_b = 0;
	cudaStat = cudaMallocPitch((void**)&d_b, &pitch_b, sizeof(float) * N, K);	
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %dx%d 2d matrix using routine cudaMallocPitch(...), file: %s, line: %d\n", N, K, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	int pitch_c = 0;
	cudaStat = cudaMallocPitch((void**)&d_c, &pitch_c, sizeof(float) * M, K);	
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %dx%d 2d matrix using routine cudaMallocPitch(...), file: %s, line: %d\n", M, K, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
#else 
	cudaStat = cudaMalloc((void**)&d_a, nbts_a);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes using routine cudaMalloc(...), file: %s, line: %d\n", nbts_a, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaStat = cudaMalloc((void**)&d_b, nbts_b);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes using routine cudaMalloc(...), file: %s, line: %d\n", nbts_b, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaStat = cudaMalloc((void**)&d_c, nbts_c);
	if(cudaStat != cudaSuccess) {
		fprintf(stderr, "ERROR: cannot allocate %ld bytes using routine cudaMalloc(...), file: %s, line: %d\n", nbts_c, __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
#endif

	stat = cublasCreate(&handle);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: CUBLAS initialization failed, file: %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaEvent_t start_load_inc;
	cudaEvent_t finish_load_inc;
	cudaEvent_t start;
	cudaEvent_t finish;
	float time;
	float time_load;
	float ops = 2.0 * M * N * K;

	cudaEventCreate(&start_load_inc);
	cudaEventCreate(&finish_load_inc);

	cudaEventRecord(start_load_inc,0);

#if defined(USE_MALLOC_PITCH)		
	stat = cublasSetMatrix(M, N, sizeof(float), h_a, M, d_a, pitch_a / sizeof(float));
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot set matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	stat = cublasSetMatrix(N, K, sizeof(float), h_b, N, d_b, pitch_b / sizeof(float));
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot set matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	float alpha = 1.f;
	float beta = 0.f;
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			   M, N, K, 
			   &alpha, 
			   d_a, pitch_a / sizeof(float), 
			   d_b, pitch_b / sizeof(float),
			   &beta,  
			   d_c, pitch_c / sizeof(float));
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot mutiply two matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(finish);
	
	cudaEventElapsedTime(&time, start, finish);
	
	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	stat = cublasGetMatrix(M, K, sizeof(float), d_c, pitch_c / sizeof(float), h_c, M);

	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot get matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
#else
	
	stat = cublasSetMatrix(M, N, sizeof(float), h_a, M, d_a, M);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot set matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	stat = cublasSetMatrix(N, K, sizeof(float), h_b, N, d_b, N);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot set matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	float alpha = 1.f;
	float beta = 0.f;
	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha, d_a, M, d_b, N, &beta, d_c, M);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot mutiply two matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(finish);
	
	cudaEventElapsedTime(&time, start, finish);
	
	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	stat = cublasGetMatrix(M, K, sizeof(float), d_c, M, h_c, M);

	if(stat != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "ERROR: cannot get matrix using routine cublasSetMatrix(...), file, %s, line: %d\n", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

#endif
	cudaEventRecord(finish_load_inc, 0);

	cudaEventSynchronize(finish_load_inc);
	
	cudaEventElapsedTime(&time_load, start_load_inc, finish_load_inc);
	
	cudaEventDestroy(start_load_inc);
	cudaEventDestroy(finish_load_inc);
/*
	cout << "Matrix C: \n";	
	for(int i = 0; i < M; i++) {
		float * c_ptr = &h_c[i];
		for(int j = 0; j < K; j++) {
			if(j < K - 1) {
				fprintf(stdout, "%.7e\t", *c_ptr);
			} else {
				fprintf(stdout, "%.7e\n", *c_ptr);
			}
			c_ptr += M;
		}
	}
*/
	cout << "Elapsed time (without load time): " << time * 0.001 << "s\n";
	cout << "Elapsed time (with load time): " << time_load * 0.001 << "s\n";
	cout << "performance: " << ops / (1000. * 1000. * 1000. * time * 0.001) << "GFLOPS\n";

#if defined(USE_PAGE_LOCK_MEMORY)
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);	
#else
	host_flatten_free_aligned<float>(h_a);
	host_flatten_free_aligned<float>(h_b);
	host_flatten_free_aligned<float>(h_c);
#endif

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	cublasDestroy(handle);

	return 0;
}

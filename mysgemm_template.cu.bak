#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

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

//#define VERBOSITY
using std::cout;

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define nullptr NULL

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

#define COL_MAJOR

inline void __safeCall(cudaError err, const char * file, const int line)
{
	if(cudaSuccess != err) {
		fprintf(stderr, "ERROR: safeCall() Runtime API error in file <%s>, line %i : %s.\n", file , line, cudaGetErrorString(err));
		exit(-1);
	}
}


class TimerGPU {
public:
	cudaEvent_t start, stop;
	cudaStream_t stream;
	TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, stream);
	}
	~TimerGPU() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	float read() {
		cudaEventRecord(stop, stream);
		cudaEventSynchronize(stop);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}
};

class TimerCPU {
	static const int bits = 10;
public:
	long long beg_clock;
	float freq;
	TimerCPU(float freq_) : freq(freq_) { 
		beg_clock = getTSC(bits);
	}
	long long getTSC(int bits) {
#ifdef WIN32
		return __rdtsc();
#else
		unsigned int low, high;
		__asm__(".byte 0x0f, 0x31" :"=a" (low), "=d" (high));
		return ((long long)high<<(32 - bits)) | ((long long)low >> bits);
#endif
	}
	float read() {
		long long end_clock = getTSC(bits);
		long long Kcycles = end_clock - beg_clock;
		float time = (float)(1 << bits) * Kcycles / freq / 1e3f;
		return time;
	}
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

template<size_t BX, size_t BY>
class CudaMatrix {
public:
	CudaMatrix();
	~CudaMatrix();
	void allocate(const int M_, const int N_, bool host, float * devmem, float * hostmem);
	double download();
	double readback();
public:
	int M, N;
	int padM, padN;
	float * h_data;
	float * d_data;
	bool h_internalAlloc;
	bool d_internalAlloc;
};

int iDivUp(int a, int b) { return (a % b == 0) ? (a / b) : (a / b + 1); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b == 0) ? a : (a - a % b + b); }
int iAlignDown(int a, int b) { return a - a % b; }

template<size_t BX, size_t BY>
void CudaMatrix<BX, BY>::allocate(const int M_, const int N_, bool host, float * devmem, float * hostmem)
{
	M = M_;
	N = N_;
	padM = iAlignUp(M, BY);
	padN = iAlignUp(N, BX);

	h_data = hostmem;
	d_data = devmem;
	if(d_data == nullptr) {
		long int nbts = sizeof(float) * (long)padM * padN;
		if(nbts < 0) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
			d_data = nullptr;
			exit(EXIT_FAILURE);
		}
		safeCall(cudaMalloc((void**)&d_data, nbts)); 
		safeCall(cudaMemset(d_data, 0, nbts));
		if(d_data == nullptr) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from device global memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
		}
		d_internalAlloc = true;
	}
	if(host && h_data == nullptr) {
		long int nbts = sizeof(float) * (long)M * N;
		if(nbts < 0) {
			fprintf(stderr, "ERROR: cannot allocate %lld bytes from host memory, file: %s, line: %d\n", nbts, __FILE__, __LINE__);
			h_data = nullptr;
			exit(EXIT_FAILURE);
		}
		h_data = (float*)malloc(nbts);
		memset(h_data, 0, nbts);
		h_internalAlloc = true;
	}
}

template<size_t BX, size_t BY>
CudaMatrix<BX, BY>::CudaMatrix() : M(0), N(0), h_data(nullptr), d_data(nullptr), h_internalAlloc(false), d_internalAlloc(false) 
{

}

template<size_t BX, size_t BY>
CudaMatrix<BX, BY>::~CudaMatrix()
{
	if(h_internalAlloc && h_data != nullptr) free(h_data);
	h_data = nullptr;
	if(d_internalAlloc && d_data != nullptr) safeCall(cudaFree(d_data));
	d_data = nullptr;
}

template<size_t BX, size_t BY>
double CudaMatrix<BX, BY>::download()
{
	TimerGPU timer(0);
#if defined(ROW_MAJOR)
	int p = sizeof(float) * padN;
	if(h_data != nullptr && d_data != nullptr) {
		safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice));
	}
#elif defined(COL_MAJOR)
	int p = sizeof(float) * padM;
	if(h_data != nullptr && d_data != nullptr) {
		safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * M, sizeof(float) * M, N, cudaMemcpyHostToDevice));
	}
#endif
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	
}

template<size_t BX, size_t BY>
double CudaMatrix<BX, BY>::readback()
{
	TimerGPU timer(0);
#if defined(ROW_MAJOR)
	int p = sizeof(float) * padN;
	safeCall(cudaMemcpy2D(h_data, sizeof(float) * N, d_data, p, sizeof(float) * N, M, cudaMemcpyDeviceToHost));
#elif defined(COL_MAJOR)
	int p = sizeof(float) * padM;
	safeCall(cudaMemcpy2D(h_data, sizeof(float) * M, d_data, p, sizeof(float) * M, N, cudaMemcpyDeviceToHost));
#endif
	double gpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: readback time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;
}

// cache A and cache B, registers prefetching
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY, 
	 size_t AX, size_t AY, size_t BX, size_t BY>
__global__ void mysgemm_cache_AB_prefetching(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	
	int idt = idy * TX + idx;
	
	int idxA = idt % AX;
	int idyA = idt / AX;

	int idxB = idt % BX;
	int idyB = idt / BX;
	
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	
	__shared__ float A_smem[BK][BM + 1];
	__shared__ float B_smem[BN][BK + 1];

	float reg_C[BN / TY][BM / TX];
	float reg_A[BM / TX];
	float reg_B[BN / TY];
	
	float reg_a[BK / AY][BM / AX];
	float reg_b[BN / BY][BK / BX];
	
	const float * daptr = dA + blx * BM + idyA * lda + idxA;
	const float * dbptr = dB + bly * BN * ldb + idyB * ldb + idxB;
	float * dcptr = dC + bly * BN * ldc + blx * BM + idy * ldc + idx;	

	int m, n, k, kk;
	
	// assume: BK % AY == 0, BM % AX == 0, BN % BY == 0, BK % BX == 0
	#pragma unroll
	for(n = 0; n < BN / TY; n++) 
		#pragma unroll
		for(m = 0; m < BM / TX; m++) {
			reg_C[n][m] = 0.f;
	}

	#pragma unroll
	for(n = 0; n < BK; n += AY) 
		#pragma unroll
		for(m = 0; m < BM; m += AX) {
			A_smem[idyA + n][idxA + m] = daptr[n * lda + m];
	}

	#pragma unroll
	for(n = 0; n < BN; n += BY)
		#pragma unroll
		for(m = 0; m < BK; m += BX) {
			B_smem[idyB + n][idxB + m] = dbptr[n * ldb + m];
	}
	__syncthreads();

	for(kk = 0; kk < ldb; kk += BK) {
		daptr += BK * lda;
		dbptr += BK;

		if(kk < ldb - BK) {		
		#pragma unroll
		for(n = 0; n < BK / AY; n++) 
			#pragma unroll
			for(m = 0; m < BM / AX; m++) {
				reg_a[n][m] = daptr[n * AY * lda + m * AX];
		}
		
		#pragma unroll
		for(n = 0; n < BN / BY; n++) 
			#pragma unroll
			for(m = 0; m < BK / BX; m++) {
				reg_b[n][m] = dbptr[n * BY * ldb + m * BX];
		}
		}

		#pragma unroll
		for(k = 0; k < BK; k++) {
			#pragma unroll
			for(m = 0; m < BM / TX; m++) {
				reg_A[m] = A_smem[k][m * TX + idx];	
			}

			#pragma unroll
			for(n = 0; n < BN / TY; n++) {
				reg_B[n] = B_smem[n * TY + idy][k];
			}	
			
			#pragma unroll
			for(n = 0; n < BN / TY; n++) 
				#pragma unroll
				for(m = 0; m < BM / TX; m++) {
					reg_C[n][m] += reg_A[m] * reg_B[n]; 
			}
		}
		__syncthreads();

		if(kk < ldb - BK) {
		#pragma unroll
		for(n = 0; n < BK / AY; n++) 
			#pragma unroll
			for(m = 0; m < BM / AX; m++) {
				A_smem[n * AY + idyA][m * AX + idxA] = reg_a[n][m];
		}

		#pragma unroll
		for(n = 0; n < BN / BY; n++) 
			#pragma unroll
			for(m = 0; m < BK / BX; m++) {
				B_smem[n * BY + idyB][m * BX + idxB] = reg_b[n][m];
		}
		}
		__syncthreads();
	}

	#pragma unroll
	for(n = 0; n < BN / TY; n++) 
		#pragma unroll
		for(m = 0; m < BM / TX; m++) {
			float regC = reg_C[n][m];
			float * memC = &dcptr[n * TY * ldc + m * TX];
			*memC = beta * (*memC) + alpha * regC;
	}
}

void constantInit(float * data, long int size, float val)
{
	for(long int i = 0; i < size; i++) {
		data[i] = val;
	}
}

template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
void mygemm_wrapper(const int M, const int K, const int N, const float alpha, const float * A, const int lda, const float * B, const int ldb, const float beta, float * C, const int ldc)
{
	CudaMatrix<BK, BM> wrapperA;
	wrapperA.allocate(lda, K, false, nullptr, const_cast<float*>(A));
	wrapperA.download();
	
	CudaMatrix<BN, BK> wrapperB;
	wrapperB.allocate(ldb, N, false, nullptr, const_cast<float*>(B));
	wrapperB.download();

	CudaMatrix<BN, BM> wrapperC;
	wrapperC.allocate(ldc, N, false, nullptr, C);
	wrapperC.download();

#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix A, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperA.padM, wrapperA.padN);
	fprintf(stdout, "INFO: matrix B, size = (%dx%d), padding size = (%dx%d)\n", K, N, wrapperB.padM, wrapperB.padN);
	fprintf(stdout, "INFO: matrix C, size = (%dx%d), padding size = (%dx%d)\n", M, N, wrapperC.padM, wrapperC.padN);
#endif

	dim3 grid( wrapperA.padM / BM, wrapperC.padN / BN, 1 );
	dim3 threads( TX, TY, 1 );
	
	TimerGPU timer(0);
	mysgemm_cache_AB_prefetching<BM, BK, BN, TX, TY, DIM_XA, TX * TY / DIM_XA, DIM_XB, TX * TY / DIM_XB><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padM, wrapperB.d_data, wrapperB.padM, beta, wrapperC.d_data, wrapperC.padM);
	double gpuTime = timer.read();


	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", gpuTime);
#ifdef VERBOSITY
	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (gpuTime / 1000.0 * 1e9));
#endif
	fflush(stdout);
	
	wrapperC.readback();
}

int main(int argc, char * argv[])
{
	if(argc != 4) {
		fprintf(stderr, "USAGE: M K N\n");
		return -1;
	}

	int M = atoi(argv[1]);
	int K = atoi(argv[2]);
	int N = atoi(argv[3]);

#ifdef VERBOSITY	
	fprintf(stdout, "INFO: matrix A (MxK) multiply matrix B (KxN), result matrix C (MxN).\n");
	fprintf(stdout, "INFO: M = %d, K = %d, N = %d\n", M, K, N);
	fflush(stdout);
#endif
	
	float * h_A = (float*)malloc(sizeof(float) * M * K);
	float * h_B = (float*)malloc(sizeof(float) * K * N);
	float * h_C = (float*)malloc(sizeof(float) * M * N);
	float * h_D = (float*)malloc(sizeof(float) * M * N);

	const float valB = 0.01f;
	long int size_A = M * K;
	long int size_B = K * N;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);
	
	long int size_C = M * N;
	long int size_D = size_C;
	memset(h_C, 0, sizeof(float) * size_C);
	memset(h_D, 0, sizeof(float) * size_D);

	// warm up
	mygemm_wrapper<ROW_BLOCK_A, ROW_BLOCK_B, COL_BLOCK_C, THREAD_BLOCK_X, THREAD_BLOCK_Y>(
		M, K, N, 1.f,
		h_A, M, h_B, K, 0.f, h_C, M);
	
	TimerCPU timer(2.60 * 1000);
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, h_A, M, h_B, K, 0.0f, h_D, M);
	double cpuTime = timer.read();
#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", cpuTime);
	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (cpuTime / 1000.0 * 1e9));
#endif
	fflush(stdout);

	// test relative error
	bool correct = true;
	double eps = 1.e-6;
	for(long int i = 0; i < size_C; i++) {
		double abs_err = fabs(h_C[i] - h_D[i]);	
		double dot_length = K;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;
	
		if (rel_err > eps) {
//	  		fprintf(stderr, "ERROR: Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], h_D[i], eps);
 	   		correct = false;
		
        	}
	}
	fprintf(stdout, "%s\n", correct ? "Result = PASS" : "Result = FAIL");
	fflush(stdout);
	
	free(h_A); h_A = nullptr;
	free(h_B); h_B = nullptr;
	free(h_C); h_C = nullptr;
	free(h_D); h_D = nullptr;

	if (!correct) {
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

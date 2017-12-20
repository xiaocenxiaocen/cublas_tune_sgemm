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

// column major
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

// row major
// cache A and cache B, registers prefetching
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY, size_t AX, size_t AY, size_t BX, size_t BY>
__global__ void mysgemm_cache_AB_prefetching_(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
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
	
	__shared__ float A_smem[BM][BK + 1];
	__shared__ float B_smem[BK][BN + 1];

	float reg_C[BM / TY][BN / TX];
// pay attention
	float reg_A[BM / TY];
	float reg_B[BN / TX];
	
	float reg_a[BM / AY][BK / AX];
	float reg_b[BK / BY][BN / BX];

	const float * daptr = dA + bly * BM * lda + idyA * lda + idxA;
	const float * dbptr = dB + blx * BN + idyB * ldb + idxB;	
	float * dcptr = dC + bly * BM * ldc + blx * BN + idy * ldc + idx;	

	int m, n, k, kk;
	
	// prefetching using registers
	// assume: BM % AY == 0, BK % AX == 0, BK % BY == 0, BN % BX == 0, BM % TY == 0, BN % TX == 0
	// initializing registers for matrix C
	#pragma unroll
	for(m = 0; m < BM / TY; m++) 
		#pragma unroll
		for(n = 0; n < BN / TX; n++) {
			reg_C[m][n] = 0.f;
	}
	
	// read fist block of A and B from device memory to shared memory
	#pragma unroll
	for(m = 0; m < BM; m += AY) 
		#pragma unroll
		for(n = 0; n < BK; n += AX) {
			A_smem[idyA + m][idxA + n] = daptr[m * lda + n];
	}

	#pragma unroll
	for(m = 0; m < BK; m += BY)
		#pragma unroll
		for(n = 0; n < BN; n += BX) {
			B_smem[idyB + m][idxB + n] = dbptr[m * ldb + n];
	}
	__syncthreads();

	for(kk = 0; kk < lda; kk += BK) {
		daptr += BK;
		dbptr += BK * ldb;

		if(kk < lda - BK) {
		// read A and B from device memory to registers
		// memory access pattern is determined by parameters (AX, AY, BX, BY)
		// check if satisfies coalesced memory access 
		#pragma unroll
		for(m = 0; m < BM / AY; m++) 
			#pragma unroll
			for(n = 0; n < BK / AX; n++) {
				reg_a[m][n] = daptr[m * AY * lda + n * AX];
		}
		
		#pragma unroll
		for(m = 0; m < BK / BY; m++)
			#pragma unroll
			for(n = 0; n < BN / BX; n++) {
				reg_b[m][n] = dbptr[m * BY * ldb + n * BX];
		}
		}

		#pragma unroll
		for(k = 0; k < BK; k++) {
			// read A and B from shared memory to registers
			// access pattern is determined by parameter (TX, TY)
			// check if causes bank conflict
			#pragma unroll
			for(m = 0; m < BM / TY; m++) {
				reg_A[m] = A_smem[m * TY + idy][k];
			}

			#pragma unroll
			for(n = 0; n < BN / TX; n++) {
				reg_B[n] = B_smem[k][n * TX + idx];
			}	
			
			// compute update of C
			#pragma unroll
			for(m = 0; m < BM / TY; m++) 
				#pragma unroll
				for(n = 0; n < BN / TX; n++) {
					reg_C[m][n] += reg_A[m] * reg_B[n]; 
			}
		}
		__syncthreads();

		if(kk < lda - BK) {
		// store A and B from registers to shared memory
		// memory access pattern is determined by parameters (AX, AY, BX, BY)
		// check if causes bank conflict
		#pragma unroll
		for(m = 0; m < BM / AY; m++) 
			#pragma unroll
			for(n = 0; n < BK / AX; n++) {
				A_smem[m * AY + idyA][n * AX + idxA] = reg_a[m][n];
		}

		#pragma unroll
		for(m = 0; m < BK / BY; m++) 
			#pragma unroll
			for(n = 0; n < BN / BX; n++) {
				B_smem[m * BY + idyB][n * BX + idxB] = reg_b[m][n];
		}
		}
		__syncthreads();
	}

	#pragma unroll
	for(m = 0; m < BM / TY; m++) 
		#pragma unroll
		for(n = 0; n < BN / TX; n++) {	
			float regC = reg_C[m][n];
			float * memC = &dcptr[m * TY * ldc + n * TX];
			*memC = beta * (*memC) + alpha * regC;
	}
}

// row major
// cache A and cache B, registers prefetching, double buffering
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY, size_t AX, size_t AY, size_t BX, size_t BY>
__global__ void mysgemm_cache_AB_prefetching_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
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
	
	__shared__ float A_smem[BM][BK + 1];
	__shared__ float B_smem[BK][BN + 1];

	float reg_C[BM / TY][BN / TX];
// pay attention
	float reg_A[BM / TY];
	float reg_B[BN / TX];
	
//	float reg_a[BM / AY][BK / AX];
//	float reg_b[BK / BY][BN / BX];

	const float * daptr = dA + bly * BM * lda + idyA * lda + idxA;
	const float * dbptr = dB + blx * BN + idyB * ldb + idxB;	
	float * dcptr = dC + bly * BM * ldc + blx * BN + idy * ldc + idx;	

	int m, n, k, kk;
	
	// prefetching using registers
	// assume: BM % AY == 0, BK % (2 * AX) == 0, BK % (2 * BY) == 0, BN % BX == 0, BM % TY == 0, BN % TX == 0
	// initializing registers for matrix C
	#pragma unroll
	for(m = 0; m < BM / TY; m++) 
		#pragma unroll
		for(n = 0; n < BN / TX; n++) {
			reg_C[m][n] = 0.f;
	}
	
	// read fist block of A and B from device memory to shared memory
	#pragma unroll
	for(m = 0; m < BM; m += AY) 
		#pragma unroll
		for(n = 0; n < BK / 2; n += AX) {
			A_smem[idyA + m][idxA + n] = daptr[m * lda + n];
	}

	#pragma unroll
	for(m = 0; m < BK / 2; m += BY)
		#pragma unroll
		for(n = 0; n < BN; n += BX) {
			B_smem[idyB + m][idxB + n] = dbptr[m * ldb + n];
	}
	__syncthreads();

	for(kk = 0; kk < lda; kk += BK) {
		#pragma unroll
		for(m = 0; m < BM; m += AY) 
			#pragma unroll
			for(n = 0; n < BK / 2; n += AX) {
				A_smem[idyA + m][idxA + n + BK / 2] = daptr[m * lda + n + BK / 2];
		}
	
		#pragma unroll
		for(m = 0; m < BK / 2; m += BY)
			#pragma unroll
			for(n = 0; n < BN; n += BX) {
				B_smem[idyB + m + BK / 2][idxB + n] = dbptr[(m + BK / 2) * ldb + n];
		}

		#pragma unroll
		for(k = 0; k < BK / 2; k++) {
			// read A and B from shared memory to registers
			// access pattern is determined by parameter (TX, TY)
			// check if causes bank conflict
			#pragma unroll
			for(m = 0; m < BM / TY; m++) {
				reg_A[m] = A_smem[m * TY + idy][k];
			}

			#pragma unroll
			for(n = 0; n < BN / TX; n++) {
				reg_B[n] = B_smem[k][n * TX + idx];
			}	
			
			// compute update of C
			#pragma unroll
			for(m = 0; m < BM / TY; m++) 
				#pragma unroll
				for(n = 0; n < BN / TX; n++) {
					reg_C[m][n] += reg_A[m] * reg_B[n]; 
			}
		}
		__syncthreads();

		daptr += BK;
		dbptr += ldb * BK;

		if(kk < lda - BK) {
		#pragma unroll
		for(m = 0; m < BM; m += AY) 
			#pragma unroll
			for(n = 0; n < BK / 2; n += AX) {
				A_smem[idyA + m][idxA + n] = daptr[m * lda + n];
		}
	
		#pragma unroll
		for(m = 0; m < BK / 2; m += BY)
			#pragma unroll
			for(n = 0; n < BN; n += BX) {
				B_smem[idyB + m][idxB + n] = dbptr[m * ldb + n];
		}
		}
		
		#pragma unroll
		for(k = 0; k < BK / 2; k++) {
			// read A and B from shared memory to registers
			// access pattern is determined by parameter (TX, TY)
			// check if causes bank conflict
			#pragma unroll
			for(m = 0; m < BM / TY; m++) {
				reg_A[m] = A_smem[m * TY + idy][k + BK / 2];
			}

			#pragma unroll
			for(n = 0; n < BN / TX; n++) {
				reg_B[n] = B_smem[k + BK / 2][n * TX + idx];
			}	
			
			// compute update of C
			#pragma unroll
			for(m = 0; m < BM / TY; m++) 
				#pragma unroll
				for(n = 0; n < BN / TX; n++) {
					reg_C[m][n] += reg_A[m] * reg_B[n]; 
			}
		}
		
		__syncthreads();
	}

	#pragma unroll
	for(m = 0; m < BM / TY; m++) 
		#pragma unroll
		for(n = 0; n < BN / TX; n++) {	
			float regC = reg_C[m][n];
			float * memC = &dcptr[m * TY * ldc + n * TX];
			*memC = beta * (*memC) + alpha * regC;
	}
}

// BM = 64
// BK = 32
// BN = 96
// TX = 16
// TY = 16
// AX = 32
// AY = 8
// BX = 32
// BY = 8
__global__ void mysgemm_fermi_cache_AB_prefetching_row_major(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[64][33];
	__shared__ float B_smem[32][97];

	float reg_C[24] = {0.f};
	float reg_A[4];
	float reg_B[6];
	
	float reg_a[8] = {0.f};
	float reg_b[12] = {0.f};

	int tid = (threadIdx.y<<4) + threadIdx.x;
	
	const float * daptr = dA + ((blockIdx.y << 6) + (tid >> 5)) * lda + (tid & 0x1f);
	const float * dbptr = dB + 3 * (blockIdx.x << 5) + (tid >> 5) * ldb + (tid & 0x1f);
	float * dcptr = dC + (blockIdx.y << 6) * ldc + (blockIdx.x << 5) + threadIdx.y * ldc + threadIdx.x;

	int k, kk;
	
	float * As = A_smem[tid >> 5] + (tid & 0x1f);
	float * Bs = B_smem[tid >> 5] + (tid & 0x1f);
	
	As[0 * 264] = daptr[0 * (lda<<3)];
	As[1 * 264] = daptr[1 * (lda<<3)];
	As[2 * 264] = daptr[2 * (lda<<3)];
	As[3 * 264] = daptr[3 * (lda<<3)];
	As[4 * 264] = daptr[4 * (lda<<3)];
	As[5 * 264] = daptr[5 * (lda<<3)];
	As[6 * 264] = daptr[6 * (lda<<3)];
	As[7 * 264] = daptr[7 * (lda<<3)];

	Bs[0 * 776 + 0 * 32] = dbptr[0 * (ldb<<3) + 0 * 32];
	Bs[0 * 776 + 1 * 32] = dbptr[0 * (ldb<<3) + 1 * 32];
	Bs[0 * 776 + 2 * 32] = dbptr[0 * (ldb<<3) + 2 * 32];
	
	Bs[1 * 776 + 0 * 32] = dbptr[1 * (ldb<<3) + 0 * 32];
	Bs[1 * 776 + 1 * 32] = dbptr[1 * (ldb<<3) + 1 * 32];
	Bs[1 * 776 + 2 * 32] = dbptr[1 * (ldb<<3) + 2 * 32];
	
	Bs[2 * 776 + 0 * 32] = dbptr[2 * (ldb<<3) + 0 * 32];
	Bs[2 * 776 + 1 * 32] = dbptr[2 * (ldb<<3) + 1 * 32];
	Bs[2 * 776 + 2 * 32] = dbptr[2 * (ldb<<3) + 2 * 32];
	
	Bs[3 * 776 + 0 * 32] = dbptr[3 * (ldb<<3) + 0 * 32];
	Bs[3 * 776 + 1 * 32] = dbptr[3 * (ldb<<3) + 1 * 32];
	Bs[3 * 776 + 2 * 32] = dbptr[3 * (ldb<<3) + 2 * 32];

	__syncthreads();
	
	for(kk = 0; kk < lda; kk += 32) {
		daptr += 32;
		dbptr += (ldb<<5);
	
		if(kk < lda - 32) {	
		reg_a[0] = daptr[0 * (lda<<3)];
		reg_a[1] = daptr[1 * (lda<<3)];
		reg_a[2] = daptr[2 * (lda<<3)];
		reg_a[3] = daptr[3 * (lda<<3)];
		reg_a[4] = daptr[4 * (lda<<3)];
		reg_a[5] = daptr[5 * (lda<<3)];
		reg_a[6] = daptr[6 * (lda<<3)];
		reg_a[7] = daptr[7 * (lda<<3)];

		reg_b[ 0] = dbptr[0 * (ldb<<3) + 0 * 32];
		reg_b[ 1] = dbptr[0 * (ldb<<3) + 1 * 32];
		reg_b[ 2] = dbptr[0 * (ldb<<3) + 2 * 32];
		
		reg_b[ 3] = dbptr[1 * (ldb<<3) + 0 * 32];
		reg_b[ 4] = dbptr[1 * (ldb<<3) + 1 * 32];
		reg_b[ 5] = dbptr[1 * (ldb<<3) + 2 * 32];
		
		reg_b[ 6] = dbptr[2 * (ldb<<3) + 0 * 32];
		reg_b[ 7] = dbptr[2 * (ldb<<3) + 1 * 32];
		reg_b[ 8] = dbptr[2 * (ldb<<3) + 2 * 32];
		
		reg_b[ 9] = dbptr[3 * (ldb<<3) + 0 * 32];
		reg_b[10] = dbptr[3 * (ldb<<3) + 1 * 32];
		reg_b[11] = dbptr[3 * (ldb<<3) + 2 * 32];
		}
	
		#pragma unroll
		for(k = 0; k < 32; k++) {
			float * As = A_smem[threadIdx.y] + k;
			float * Bs = B_smem[k] + threadIdx.x;
			
			reg_A[0] = As[0 * 528];
			reg_A[1] = As[1 * 528];
			reg_A[2] = As[2 * 528];
			reg_A[3] = As[3 * 528];
			
			reg_B[0] = Bs[0 * 16];
			reg_B[1] = Bs[1 * 16];
			reg_B[2] = Bs[2 * 16];
			reg_B[3] = Bs[3 * 16];
			reg_B[4] = Bs[4 * 16];
			reg_B[5] = Bs[5 * 16];
			
			reg_C[ 0] += reg_A[0] * reg_B[0];	
			reg_C[ 1] += reg_A[0] * reg_B[1];
			reg_C[ 2] += reg_A[0] * reg_B[2];	
			reg_C[ 3] += reg_A[0] * reg_B[3];
			reg_C[ 4] += reg_A[0] * reg_B[4];	
			reg_C[ 5] += reg_A[0] * reg_B[5];
			
			reg_C[ 6] += reg_A[1] * reg_B[0];	
			reg_C[ 7] += reg_A[1] * reg_B[1];
			reg_C[ 8] += reg_A[1] * reg_B[2];	
			reg_C[ 9] += reg_A[1] * reg_B[3];
			reg_C[10] += reg_A[1] * reg_B[4];	
			reg_C[11] += reg_A[1] * reg_B[5];
		
			reg_C[12] += reg_A[2] * reg_B[0];	
			reg_C[13] += reg_A[2] * reg_B[1];
			reg_C[14] += reg_A[2] * reg_B[2];	
			reg_C[15] += reg_A[2] * reg_B[3];
			reg_C[16] += reg_A[2] * reg_B[4];	
			reg_C[17] += reg_A[2] * reg_B[5];	
		
			reg_C[18] += reg_A[3] * reg_B[0];	
			reg_C[19] += reg_A[3] * reg_B[1];
			reg_C[20] += reg_A[3] * reg_B[2];	
			reg_C[21] += reg_A[3] * reg_B[3];
			reg_C[22] += reg_A[3] * reg_B[4];	
			reg_C[23] += reg_A[3] * reg_B[5];
		}
		__syncthreads();

		if(kk < lda - 32) {
			float * As = A_smem[tid >> 5] + (tid & 0x1f);
			float * Bs = B_smem[tid >> 5] + (tid & 0x1f);

			As[0 * 264] = reg_a[0];
			As[1 * 264] = reg_a[1];
			As[2 * 264] = reg_a[2];
			As[3 * 264] = reg_a[3];
			As[4 * 264] = reg_a[4];
			As[5 * 264] = reg_a[5];
			As[6 * 264] = reg_a[6];
			As[7 * 264] = reg_a[7];

			Bs[0 * 776 + 0 * 32] = reg_b[ 0];
			Bs[0 * 776 + 1 * 32] = reg_b[ 1];
			Bs[0 * 776 + 2 * 32] = reg_b[ 2];
			                       
			Bs[1 * 776 + 0 * 32] = reg_b[ 3];
			Bs[1 * 776 + 1 * 32] = reg_b[ 4];
			Bs[1 * 776 + 2 * 32] = reg_b[ 5];
			                       
			Bs[2 * 776 + 0 * 32] = reg_b[ 6];
			Bs[2 * 776 + 1 * 32] = reg_b[ 7];
			Bs[2 * 776 + 2 * 32] = reg_b[ 8];
			                       
			Bs[3 * 776 + 0 * 32] = reg_b[ 9];
			Bs[3 * 776 + 1 * 32] = reg_b[10];
			Bs[3 * 776 + 2 * 32] = reg_b[11];
		}	
		__syncthreads();
	}

	dcptr[0 * (ldc<<4) + 0 * 16] = beta * dcptr[0 * (ldc<<4) + 0 * 16] + reg_C[ 0];
	dcptr[0 * (ldc<<4) + 1 * 16] = beta * dcptr[0 * (ldc<<4) + 1 * 16] + reg_C[ 1];
	dcptr[0 * (ldc<<4) + 2 * 16] = beta * dcptr[0 * (ldc<<4) + 2 * 16] + reg_C[ 2];
	dcptr[0 * (ldc<<4) + 3 * 16] = beta * dcptr[0 * (ldc<<4) + 3 * 16] + reg_C[ 3];
	dcptr[0 * (ldc<<4) + 4 * 16] = beta * dcptr[0 * (ldc<<4) + 4 * 16] + reg_C[ 4];
	dcptr[0 * (ldc<<4) + 5 * 16] = beta * dcptr[0 * (ldc<<4) + 5 * 16] + reg_C[ 5];
                                                                             
	dcptr[1 * (ldc<<4) + 0 * 16] = beta * dcptr[1 * (ldc<<4) + 0 * 16] + reg_C[ 6];
	dcptr[1 * (ldc<<4) + 1 * 16] = beta * dcptr[1 * (ldc<<4) + 1 * 16] + reg_C[ 7];
	dcptr[1 * (ldc<<4) + 2 * 16] = beta * dcptr[1 * (ldc<<4) + 2 * 16] + reg_C[ 8];
	dcptr[1 * (ldc<<4) + 3 * 16] = beta * dcptr[1 * (ldc<<4) + 3 * 16] + reg_C[ 9];
	dcptr[1 * (ldc<<4) + 4 * 16] = beta * dcptr[1 * (ldc<<4) + 4 * 16] + reg_C[10];
	dcptr[1 * (ldc<<4) + 5 * 16] = beta * dcptr[1 * (ldc<<4) + 5 * 16] + reg_C[11];
                                                                                      
	dcptr[2 * (ldc<<4) + 0 * 16] = beta * dcptr[2 * (ldc<<4) + 0 * 16] + reg_C[12];
	dcptr[2 * (ldc<<4) + 1 * 16] = beta * dcptr[2 * (ldc<<4) + 1 * 16] + reg_C[13];
	dcptr[2 * (ldc<<4) + 2 * 16] = beta * dcptr[2 * (ldc<<4) + 2 * 16] + reg_C[14];
	dcptr[2 * (ldc<<4) + 3 * 16] = beta * dcptr[2 * (ldc<<4) + 3 * 16] + reg_C[15];
	dcptr[2 * (ldc<<4) + 4 * 16] = beta * dcptr[2 * (ldc<<4) + 4 * 16] + reg_C[16];
	dcptr[2 * (ldc<<4) + 5 * 16] = beta * dcptr[2 * (ldc<<4) + 5 * 16] + reg_C[17];
                                                                                      
	dcptr[3 * (ldc<<4) + 0 * 16] = beta * dcptr[3 * (ldc<<4) + 0 * 16] + reg_C[18];
	dcptr[3 * (ldc<<4) + 1 * 16] = beta * dcptr[3 * (ldc<<4) + 1 * 16] + reg_C[19];
	dcptr[3 * (ldc<<4) + 2 * 16] = beta * dcptr[3 * (ldc<<4) + 2 * 16] + reg_C[20];
	dcptr[3 * (ldc<<4) + 3 * 16] = beta * dcptr[3 * (ldc<<4) + 3 * 16] + reg_C[21];
	dcptr[3 * (ldc<<4) + 4 * 16] = beta * dcptr[3 * (ldc<<4) + 4 * 16] + reg_C[22];
	dcptr[3 * (ldc<<4) + 5 * 16] = beta * dcptr[3 * (ldc<<4) + 5 * 16] + reg_C[23];
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
#if defined(ROW_MAJOR)
	wrapperA.allocate(M, lda, false, nullptr, const_cast<float*>(A));
#elif defined(COL_MAJOR)
	wrapperA.allocate(lda, K, false, nullptr, const_cast<float*>(A));
#endif
	wrapperA.download();
	
	CudaMatrix<BN, BK> wrapperB;
#if defined(ROW_MAJOR)
	wrapperB.allocate(K, ldb, false, nullptr, const_cast<float*>(B));
#elif defined(COL_MAJOR)
	wrapperB.allocate(ldb, N, false, nullptr, const_cast<float*>(B));
#endif
	wrapperB.download();

	CudaMatrix<BN, BM> wrapperC;
#if defined(ROW_MAJOR)
	wrapperC.allocate(M, ldc, false, nullptr, C);
#elif defined(COL_MAJOR)
	wrapperC.allocate(ldc, N, false, nullptr, C);
#endif
	wrapperC.download();

#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix A, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperA.padM, wrapperA.padN);
	fprintf(stdout, "INFO: matrix B, size = (%dx%d), padding size = (%dx%d)\n", K, N, wrapperB.padM, wrapperB.padN);
	fprintf(stdout, "INFO: matrix C, size = (%dx%d), padding size = (%dx%d)\n", M, N, wrapperC.padM, wrapperC.padN);
#endif

#if defined(ROW_MAJOR)
	dim3 grid( wrapperC.padN / BN, wrapperA.padM / BM, 1 );
#elif defined(COL_MAJOR)
	dim3 grid( wrapperA.padM / BM, wrapperC.padN / BN, 1 );
#endif
	dim3 threads( TX, TY, 1 );
	
	TimerGPU timer(0);
#if defined(ROW_MAJOR)
	mysgemm_cache_AB_prefetching_<BM, BK, BN, TX, TY, DIM_XA, TX * TY / DIM_XA, DIM_XB, TX * TY / DIM_XB><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
//	mysgemm_cache_AB_prefetching_double_buffering<BM, BK, BN, TX, TY, DIM_XA, TX * TY / DIM_XA, DIM_XB, TX * TY / DIM_XB><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
//	mysgemm_fermi_cache_AB_prefetching_row_major<<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
#elif defined(COL_MAJOR)
	mysgemm_cache_AB_prefetching<BM, BK, BN, TX, TY, DIM_XA, TX * TY / DIM_XA, DIM_XB, TX * TY / DIM_XB><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padM, wrapperB.d_data, wrapperB.padM, beta, wrapperC.d_data, wrapperC.padM);
#endif
	double gpuTime = timer.read();

	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (gpuTime / 1000.0 * 1e9));
#ifdef VERBOSITY
	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", gpuTime);
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
	#if defined(ROW_MAJOR)
	mygemm_wrapper<ROW_BLOCK_A, ROW_BLOCK_B, COL_BLOCK_C, THREAD_BLOCK_X, THREAD_BLOCK_Y>(
		M, K, N, 1.f,
		h_A, K, h_B, N, 0.f, h_C, N);
	#elif defined(COL_MAJOR)
	mygemm_wrapper<ROW_BLOCK_A, ROW_BLOCK_B, COL_BLOCK_C, THREAD_BLOCK_X, THREAD_BLOCK_Y>(
		M, K, N, 1.f,
		h_A, M, h_B, K, 0.f, h_C, M);
	#endif
	
	TimerCPU timer(2.60 * 1000);
	#if defined(ROW_MAJOR)
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, h_A, K, h_B, N, 0.0f, h_D, N);
	#elif defined(COL_MAJOR)
	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, h_A, M, h_B, K, 0.0f, h_D, M);
	#endif
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

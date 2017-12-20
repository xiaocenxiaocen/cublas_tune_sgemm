#include <stdio.h>
#include <mkl.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

#define VERBOSE
using std::cout;

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define nullptr NULL

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)

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
	int p = sizeof(float) * padN;
	if(h_data != nullptr && d_data != nullptr) {
		safeCall(cudaMemcpy2D(d_data, p, h_data, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice));
	}
	double gpuTime = timer.read();
#ifdef VERBOSE
	fprintf(stdout, "INFO: download time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;	
}

template<size_t BX, size_t BY>
double CudaMatrix<BX, BY>::readback()
{
	TimerGPU timer(0);
	int p = sizeof(float) * padN;
//	cout << sizeof(float) * N << "\t" << p << "\n";
//	if(h_data == nullptr) cout << "1\n";
//	if(d_data == nullptr) cout << "2\n";
	safeCall(cudaMemcpy2D(h_data, sizeof(float) * N, d_data, p, sizeof(float) * N, M, cudaMemcpyDeviceToHost));
	double gpuTime = timer.read();
#ifdef VERBOSE
	fprintf(stdout, "INFO: readback time = %.2fms\n", gpuTime);
	fflush(stdout);
#endif
	return gpuTime;
}

// cache A and cache B
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		const float * dbptr_ = dbptr + tidy * ldb; 
		for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		__syncthreads();
		
		for(int kk = 0; kk < BK; kk++) {
			#pragma unroll
			for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
				#pragma unroll
				for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	#pragma unroll
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		#pragma unroll
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache A and cache B and prefetching
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB_prefetching(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
	float A_reg[BM / TY][BK / TX];
	float B_reg[BK / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	// load block of A to shared memory
	const float * daptr_ = daptr + tidy * lda;
	for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
		for(int ij = tidx; ij < BK; ij += TX) {
			A_smem[ii][ij] = daptr_[ij];
		}
	}

	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		if(ik < lda - 1) {
		// load block of A to registers
		const float * daptr_ = daptr + tidy * lda + BK;
		for(int ii = tidy, _ii = 0; ii < BM; ii += TY, _ii++, daptr_ += TY * lda) {
			for(int ij = tidx, _ij = 0; ij < BK; ij += TX, _ij++) {
				A_reg[_ii][_ij] = daptr_[ij];
			}
		}

		// load block of B to registers
		const float * dbptr_ = dbptr + tidy * ldb + stride_b; 
		for(int ii = tidy, _ii = 0; ii < BK; ii += TY, _ii++, dbptr_ += TY * ldb) {
			for(int ij = tidx, _ij = 0; ij < BN; ij += TX, _ij++) {
				B_reg[_ii][_ij] = dbptr_[ij];
			}
		}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				#pragma unroll
				for(int kk = 0; kk < BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
//					C_reg[ii][ij] += A_reg[kk] * B_smem[kk][in];
				}
			}
		}

		if(ik < lda - 1) {
		// store registers to A_smem
		for(int ii = tidy, _ii = 0; ii < BM; ii += TY, _ii++) {
			for(int ij = tidx, _ij = 0; ij < BK; ij += TX, _ij++) {
				A_smem[ii][ij] = A_reg[_ii][_ij];
			}
		}

		// store registers to B_smem
		for(int ii = tidy, _ii = 0; ii < BK; ii += TY, _ii++) {
			for(int ij = tidx, _ij = 0; ij < BN; ij += TX, _ij++) {
				B_smem[ii][ij] = B_reg[_ii][_ij];
			}
		}
		}
		
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache A and cache B and double-buffering
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_AB_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	const int HALF_BK = BK / 2;

	const float * daptr_ = daptr + tidy * lda;
	for(int ii = tidy; ii < BM ; ii += TY, daptr_ += TY * lda) {
		for(int ij = tidx; ij < HALF_BK; ij += TX) {
			A_smem[ii][ij] = daptr_[ij];
		}
	}

	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();

	for(int ik = 0; ik < lda; ik += BK) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = HALF_BK + tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		const float * dbptr_ = dbptr + (HALF_BK + tidy) * ldb; 
		for(int ii = HALF_BK + tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = 0; kk < HALF_BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();

		daptr += BK, dbptr += stride_b;
		if(ik < lda - 1) {
			// load block of A to shared memory
			daptr_ = daptr + tidy * lda;
			for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
				for(int ij = tidx; ij < HALF_BK; ij += TX) {
					A_smem[ii][ij] = daptr_[ij];
				}
			}

			dbptr_ = dbptr + tidy * ldb; 
			for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
				for(int ij = tidx; ij < BN; ij += TX) {
					B_smem[ii][ij] = dbptr_[ij];
				}
			}
		}
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = HALF_BK; kk < BK; kk++) {
					C_reg[ii][ij] += A_smem[im][kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache B
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_B(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
	float A_reg[BK];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	#pragma unroll
	for(int ii = 0; ii < BM / TY; ii++) {
		#pragma unroll
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of B to shared memory
		const float * dbptr_ = dbptr + tidy * ldb; 
		#pragma unroll
		for(int ii = tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			#pragma unroll
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		__syncthreads();

		const float * daptr_ = daptr + tidy * lda;		
		#pragma unroll
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++, daptr_ += TY * lda) {
			#pragma unroll
			for(int kk = 0; kk < BK; kk++) {
				A_reg[kk] = daptr_[kk];
			}
			#pragma unroll
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				#pragma unroll
				for(int kk = 0; kk < BK; kk++) {
					C_reg[ii][ij] += A_reg[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	#pragma unroll
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		#pragma unroll
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache B and double buffering
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_B_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[BK][BN];
	float C_reg[BM / TY][BN / TX];
//	float A_reg[BK];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	const int HALF_BK = BK / 2;

	// load block of B to shared memory
	const float * dbptr_ = dbptr + tidy * ldb; 
	for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
		for(int ij = tidx; ij < BN; ij += TX) {
			B_smem[ii][ij] = dbptr_[ij];
		}
	}
	__syncthreads();
	
	for(int ik = 0; ik < lda; ik += BK) {
		// load block of B to shared memory
		const float * dbptr_ = dbptr + (HALF_BK + tidy) * ldb; 
		for(int ii = HALF_BK + tidy; ii < BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		
		const float * daptr_ = daptr + tidy * lda;		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++, daptr_ += TY * lda) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = 0; kk < HALF_BK; kk++) {
					C_reg[ii][ij] += daptr_[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();

		daptr += BK, dbptr += stride_b;
	
		if(ik < lda - 1) {
		// load block of B to shared memory
		dbptr_ = dbptr + tidy * ldb; 
		for(int ii = tidy; ii < HALF_BK; ii += TY, dbptr_ += TY * ldb) {
			for(int ij = tidx; ij < BN; ij += TX) {
				B_smem[ii][ij] = dbptr_[ij];
			}
		}
		}
		
		daptr_ = daptr + tidy * lda - BK;		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++, daptr_ += TY * lda) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				for(int kk = HALF_BK; kk < BK; kk++) {
					C_reg[ii][ij] += daptr_[kk] * B_smem[kk][in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

// cache A
template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
__global__ void mysgemm_cache_A(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float A_smem[BM][BK];
//	__shared__ float B_smem[BK][BN];
//	__shared__ float B_smem[BN];
	float C_reg[BM / TY][BN / TX];

	const int gy = blockIdx.y * BM;
	const int gx = blockIdx.x * BN;
	
	const int tidy = threadIdx.y;
	const int tidx = threadIdx.x;
	
	const float * daptr = dA + gy * lda;
	const float * dbptr = dB + gx;
	float * dcptr = dC + gy * ldc + gx;
	
	const int stride_b = BK * ldb;

	for(int ii = 0; ii < BM / TY; ii++) {
		for(int ij = 0; ij < BN / TX; ij++) {
			C_reg[ii][ij] = 0.f;
		}
	}

	for(int ik = 0; ik < lda; ik += BK, daptr += BK, dbptr += stride_b) {
		// load block of A to shared memory
		const float * daptr_ = daptr + tidy * lda;
		for(int ii = tidy; ii < BM; ii += TY, daptr_ += TY * lda) {
			for(int ij = tidx; ij < BK; ij += TX) {
				A_smem[ii][ij] = daptr_[ij];
			}
		}

		__syncthreads();
		
		for(int im = tidy, ii = 0; im < BM; im += TY, ii++) {
			for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
				const float * dbptr_ = dbptr;
				for(int kk = 0; kk < BK; kk++, dbptr_ += ldb) {
//					C_reg[ii][ij] += A_smem[im][kk] * B_smem[in];
					C_reg[ii][ij] += A_smem[im][kk] * dbptr_[in];
				}
			}
		}
		__syncthreads();
	}

	float * dcptr_ = dcptr + tidy * ldc;
	for(int im = tidy, ii = 0; im < BM; im += TY, dcptr_ += TY * ldc, ii++) {
		for(int in = tidx, ij = 0; in < BN; in += TX, ij++) {
			dcptr_[in] = beta * dcptr_[in] + alpha * C_reg[ii][ij];
		}
	}
}

__device__ float s_dot16(float * a, float * bs)
{
	float ret = 0.f;
	ret += a[ 0] * bs[ 0 * 64];
	ret += a[ 1] * bs[ 1 * 64];
	ret += a[ 2] * bs[ 2 * 64];
	ret += a[ 3] * bs[ 3 * 64];
	ret += a[ 4] * bs[ 4 * 64];
	ret += a[ 5] * bs[ 5 * 64];
	ret += a[ 6] * bs[ 6 * 64];
	ret += a[ 7] * bs[ 7 * 64];
	ret += a[ 8] * bs[ 8 * 64];
	ret += a[ 9] * bs[ 9 * 64];
	ret += a[10] * bs[10 * 64];
	ret += a[11] * bs[11 * 64];
	ret += a[12] * bs[12 * 64];
	ret += a[13] * bs[13 * 64];
	ret += a[14] * bs[14 * 64];
	ret += a[15] * bs[15 * 64];
	return ret;
}
__global__ void mysgemm_cache_B_unrolling_v1(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float B_smem[1024];
	float C_reg[4] = {0.f};
	float A_reg[16] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<4) + threadIdx.y) * lda;
	const float *  dbptr = dB + (blockIdx.x<<6) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<4) + threadIdx.y) * ldc + (blockIdx.x<<6) + threadIdx.x;
	
	for(int ik = 0; ik < lda; ik += 16) {
		float * Bs = &B_smem[(threadIdx.y<<6) + threadIdx.x];
		Bs[0 * 16] = dbptr[0 * 16];
		Bs[1 * 16] = dbptr[1 * 16];		
		Bs[2 * 16] = dbptr[2 * 16];
		Bs[3 * 16] = dbptr[3 * 16];
		__syncthreads();
					
		A_reg[ 0] = daptr[ 0];
		A_reg[ 1] = daptr[ 1];	
		A_reg[ 2] = daptr[ 2];
		A_reg[ 3] = daptr[ 3];
		A_reg[ 4] = daptr[ 4];
		A_reg[ 5] = daptr[ 5];
		A_reg[ 6] = daptr[ 6];
		A_reg[ 7] = daptr[ 7];
		A_reg[ 8] = daptr[ 8];
		A_reg[ 9] = daptr[ 9];	
		A_reg[10] = daptr[10];
		A_reg[11] = daptr[11];
		A_reg[12] = daptr[12];
		A_reg[13] = daptr[13];
		A_reg[14] = daptr[14];
		A_reg[15] = daptr[15];

		Bs = &B_smem[threadIdx.x];
		C_reg[0] += s_dot16(A_reg, &Bs[ 0 * 16]);
		C_reg[1] += s_dot16(A_reg, &Bs[ 1 * 16]);
		C_reg[2] += s_dot16(A_reg, &Bs[ 2 * 16]);
		C_reg[3] += s_dot16(A_reg, &Bs[ 3 * 16]);	
		
		__syncthreads();
		daptr += 16;
		dbptr += (ldb<<4);
	}
	
	dcptr[0 * 16] = beta * dcptr[0 * 16] + alpha * C_reg[0];
	dcptr[1 * 16] = beta * dcptr[1 * 16] + alpha * C_reg[1];
	dcptr[2 * 16] = beta * dcptr[2 * 16] + alpha * C_reg[2];
	dcptr[3 * 16] = beta * dcptr[3 * 16] + alpha * C_reg[3];
}

__global__ void mysgemm_ilp(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[4] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda + threadIdx.x;
	const float *  dbptr = dB + (blockIdx.x<<5) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<5) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 32) {
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
		
		As[0 * (1<<8)] = daptr[0 * (lda << 3)];
		As[1 * (1<<8)] = daptr[1 * (lda << 3)];
		As[2 * (1<<8)] = daptr[2 * (lda << 3)];
		As[3 * (1<<8)] = daptr[3 * (lda << 3)];

		Bs[0 * (1<<8)] = dbptr[0 * (ldb << 3)];
		Bs[1 * (1<<8)] = dbptr[1 * (ldb << 3)];
		Bs[2 * (1<<8)] = dbptr[2 * (ldb << 3)];
		Bs[3 * (1<<8)] = dbptr[3 * (ldb << 3)];

		__syncthreads();
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<8) + kk] * Bs[(kk<<5)];
			C_reg[1] += As[1 * (1<<8) + kk] * Bs[(kk<<5)];
			C_reg[2] += As[2 * (1<<8) + kk] * Bs[(kk<<5)];
			C_reg[3] += As[3 * (1<<8) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();				

		daptr += 32;
		dbptr += (ldb << 5);
	}
	
	dcptr[0 * (ldc<<3)] = beta * dcptr[0 * (ldc<<3)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<3)] = beta * dcptr[1 * (ldc<<3)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<3)] = beta * dcptr[2 * (ldc<<3)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<3)] = beta * dcptr[3 * (ldc<<3)] + alpha * C_reg[3];
}

__global__ void mysgemm_ilp_v2(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[8] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda + threadIdx.x;
	const float *  dbptr = dB + (blockIdx.x<<5) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<5) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 32) {
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
		
		As[0 * (1<<7)] = daptr[0 * (lda << 2)];
		As[1 * (1<<7)] = daptr[1 * (lda << 2)];
		As[2 * (1<<7)] = daptr[2 * (lda << 2)];
		As[3 * (1<<7)] = daptr[3 * (lda << 2)];
		As[4 * (1<<7)] = daptr[4 * (lda << 2)];
		As[5 * (1<<7)] = daptr[5 * (lda << 2)];
		As[6 * (1<<7)] = daptr[6 * (lda << 2)];
		As[7 * (1<<7)] = daptr[7 * (lda << 2)];
		
		Bs[0 * (1<<7)] = dbptr[0 * (ldb << 2)];
		Bs[1 * (1<<7)] = dbptr[1 * (ldb << 2)];
		Bs[2 * (1<<7)] = dbptr[2 * (ldb << 2)];
		Bs[3 * (1<<7)] = dbptr[3 * (ldb << 2)];
		Bs[4 * (1<<7)] = dbptr[4 * (ldb << 2)];
		Bs[5 * (1<<7)] = dbptr[5 * (ldb << 2)];
		Bs[6 * (1<<7)] = dbptr[6 * (ldb << 2)];
		Bs[7 * (1<<7)] = dbptr[7 * (ldb << 2)];
		
		__syncthreads();
		
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();				

		daptr += 32;
		dbptr += (ldb << 5);
	}
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[7 * (ldc<<2)] + alpha * C_reg[7];
}

__global__ void mysgemm_ilp_v2_double_buffering(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[8] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda + threadIdx.x;
	const float *  dbptr = dB + (blockIdx.x<<5) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<5) + threadIdx.x;

	float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
	float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
	As[0 * (1<<7)] = daptr[0 * (lda << 2)];
	As[1 * (1<<7)] = daptr[1 * (lda << 2)];
	As[2 * (1<<7)] = daptr[2 * (lda << 2)];
	As[3 * (1<<7)] = daptr[3 * (lda << 2)];

	Bs[0 * (1<<7)] = dbptr[0 * (ldb << 2)];
	Bs[1 * (1<<7)] = dbptr[1 * (ldb << 2)];
	Bs[2 * (1<<7)] = dbptr[2 * (ldb << 2)];
	Bs[3 * (1<<7)] = dbptr[3 * (ldb << 2)];
	__syncthreads();
	
	for(int ik = 0; ik < lda; ik += 32) {
		
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
		
		As[4 * (1<<7)] = daptr[4 * (lda << 2)];
		As[5 * (1<<7)] = daptr[5 * (lda << 2)];
		As[6 * (1<<7)] = daptr[6 * (lda << 2)];
		As[7 * (1<<7)] = daptr[7 * (lda << 2)];
		
		Bs[4 * (1<<7)] = dbptr[4 * (ldb << 2)];
		Bs[5 * (1<<7)] = dbptr[5 * (ldb << 2)];
		Bs[6 * (1<<7)] = dbptr[6 * (ldb << 2)];
		Bs[7 * (1<<7)] = dbptr[7 * (ldb << 2)];
		
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();				
		
		daptr += 32;
		dbptr += (ldb<<5);

		if(ik < lda - 32) {
			float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
			float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
			As[0 * (1<<7)] = daptr[0 * (lda << 2)];
			As[1 * (1<<7)] = daptr[1 * (lda << 2)];
			As[2 * (1<<7)] = daptr[2 * (lda << 2)];
			As[3 * (1<<7)] = daptr[3 * (lda << 2)];
		
			Bs[0 * (1<<7)] = dbptr[0 * (ldb << 2)];
			Bs[1 * (1<<7)] = dbptr[1 * (ldb << 2)];
			Bs[2 * (1<<7)] = dbptr[2 * (ldb << 2)];
			Bs[3 * (1<<7)] = dbptr[3 * (ldb << 2)];	
		}
		
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();
	}
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[7 * (ldc<<2)] + alpha * C_reg[7];
}

__global__ void mysgemm_ilp_v2_prefetching(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[8] = {0.f};
	float A_reg[8] = {0.f};
	float B_reg[8] = {0.f};	

	const float *  daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda + threadIdx.x;
	const float *  dbptr = dB + (blockIdx.x<<5) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<5) + threadIdx.x;

	float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
	float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];
	
	As[0 * (1<<7)] = daptr[0 * (lda << 2)];
	As[1 * (1<<7)] = daptr[1 * (lda << 2)];
	As[2 * (1<<7)] = daptr[2 * (lda << 2)];
	As[3 * (1<<7)] = daptr[3 * (lda << 2)];
	As[4 * (1<<7)] = daptr[4 * (lda << 2)];
	As[5 * (1<<7)] = daptr[5 * (lda << 2)];
	As[6 * (1<<7)] = daptr[6 * (lda << 2)];
	As[7 * (1<<7)] = daptr[7 * (lda << 2)];
	
	Bs[0 * (1<<7)] = dbptr[0 * (ldb << 2)];
	Bs[1 * (1<<7)] = dbptr[1 * (ldb << 2)];
	Bs[2 * (1<<7)] = dbptr[2 * (ldb << 2)];
	Bs[3 * (1<<7)] = dbptr[3 * (ldb << 2)];
	Bs[4 * (1<<7)] = dbptr[4 * (ldb << 2)];
	Bs[5 * (1<<7)] = dbptr[5 * (ldb << 2)];
	Bs[6 * (1<<7)] = dbptr[6 * (ldb << 2)];
	Bs[7 * (1<<7)] = dbptr[7 * (ldb << 2)];
		
	__syncthreads();
	
	for(int ik = 0; ik < lda; ik += 32) {
		
		if(ik < lda - 32) {
		daptr += 32;
		dbptr += (ldb << 5);
		A_reg[0] = daptr[0 * (lda << 2)];
		A_reg[1] = daptr[1 * (lda << 2)];
		A_reg[2] = daptr[2 * (lda << 2)];
		A_reg[3] = daptr[3 * (lda << 2)];
		A_reg[4] = daptr[4 * (lda << 2)];
		A_reg[5] = daptr[5 * (lda << 2)];
		A_reg[6] = daptr[6 * (lda << 2)];
		A_reg[7] = daptr[7 * (lda << 2)];
		
		B_reg[0] = dbptr[0 * (ldb << 2)];
		B_reg[1] = dbptr[1 * (ldb << 2)];
		B_reg[2] = dbptr[2 * (ldb << 2)];
		B_reg[3] = dbptr[3 * (ldb << 2)];
		B_reg[4] = dbptr[4 * (ldb << 2)];
		B_reg[5] = dbptr[5 * (ldb << 2)];
		B_reg[6] = dbptr[6 * (ldb << 2)];
		B_reg[7] = dbptr[7 * (ldb << 2)];
		}		

		__syncthreads();
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();				

		if(ik < lda - 32) {
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<5) + threadIdx.x];

		As[0 * (1<<7)] = A_reg[0];
		As[1 * (1<<7)] = A_reg[1];
		As[2 * (1<<7)] = A_reg[2];
		As[3 * (1<<7)] = A_reg[3];
		As[4 * (1<<7)] = A_reg[4];
		As[5 * (1<<7)] = A_reg[5];
		As[6 * (1<<7)] = A_reg[6];
		As[7 * (1<<7)] = A_reg[7];
		
		Bs[0 * (1<<7)] = B_reg[0];
		Bs[1 * (1<<7)] = B_reg[1];
		Bs[2 * (1<<7)] = B_reg[2];
		Bs[3 * (1<<7)] = B_reg[3];
		Bs[4 * (1<<7)] = B_reg[4];
		Bs[5 * (1<<7)] = B_reg[5];
		Bs[6 * (1<<7)] = B_reg[6];
		Bs[7 * (1<<7)] = B_reg[7];
		}
	}
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[7 * (ldc<<2)] + alpha * C_reg[7];
}

__global__ void mysgemm_ilp_v3(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[3072];
	float C_reg[16] = {0.f};
	
	const float * daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda + threadIdx.x;
	const float * dbptr = dB + (blockIdx.x<<6) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<6) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 32) {
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<6) + threadIdx.x];
		
		As[0 * (1<<7)] = daptr[0 * (lda << 2)];
		As[1 * (1<<7)] = daptr[1 * (lda << 2)];
		As[2 * (1<<7)] = daptr[2 * (lda << 2)];
		As[3 * (1<<7)] = daptr[3 * (lda << 2)];
		As[4 * (1<<7)] = daptr[4 * (lda << 2)];
		As[5 * (1<<7)] = daptr[5 * (lda << 2)];
		As[6 * (1<<7)] = daptr[6 * (lda << 2)];
		As[7 * (1<<7)] = daptr[7 * (lda << 2)];
		
		Bs[0 * (1<<8)] = dbptr[0 * (ldb << 2)];
		Bs[1 * (1<<8)] = dbptr[1 * (ldb << 2)];
		Bs[2 * (1<<8)] = dbptr[2 * (ldb << 2)];
		Bs[3 * (1<<8)] = dbptr[3 * (ldb << 2)];
		Bs[4 * (1<<8)] = dbptr[4 * (ldb << 2)];
		Bs[5 * (1<<8)] = dbptr[5 * (ldb << 2)];
		Bs[6 * (1<<8)] = dbptr[6 * (ldb << 2)];
		Bs[7 * (1<<8)] = dbptr[7 * (ldb << 2)];
		
		Bs[0 * (1<<8) + 32] = dbptr[0 * (ldb << 2) + 32];
		Bs[1 * (1<<8) + 32] = dbptr[1 * (ldb << 2) + 32];
		Bs[2 * (1<<8) + 32] = dbptr[2 * (ldb << 2) + 32];
		Bs[3 * (1<<8) + 32] = dbptr[3 * (ldb << 2) + 32];
		Bs[4 * (1<<8) + 32] = dbptr[4 * (ldb << 2) + 32];
		Bs[5 * (1<<8) + 32] = dbptr[5 * (ldb << 2) + 32];
		Bs[6 * (1<<8) + 32] = dbptr[6 * (ldb << 2) + 32];
		Bs[7 * (1<<8) + 32] = dbptr[7 * (ldb << 2) + 32];
		
		__syncthreads();
		
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[1024 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<6)];
		
			C_reg[ 8] += As[0 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[ 9] += As[1 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[10] += As[2 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[11] += As[3 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[12] += As[4 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[13] += As[5 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[14] += As[6 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[15] += As[7 * (1<<7) + kk] * Bs[(kk<<6) + 32];
		}
		__syncthreads();				

		daptr += 32;
		dbptr += (ldb << 5);
	}
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[ 0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[ 1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[ 2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[ 3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[ 4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[ 5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[ 6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[ 7 * (ldc<<2)] + alpha * C_reg[7];
	dcptr[0 * (ldc<<2) + 32] = beta * dcptr[0 * (ldc<<2) + 32] + alpha * C_reg[8];
	dcptr[1 * (ldc<<2) + 32] = beta * dcptr[1 * (ldc<<2) + 32] + alpha * C_reg[9];
	dcptr[2 * (ldc<<2) + 32] = beta * dcptr[2 * (ldc<<2) + 32] + alpha * C_reg[10];
	dcptr[3 * (ldc<<2) + 32] = beta * dcptr[3 * (ldc<<2) + 32] + alpha * C_reg[11];
	dcptr[4 * (ldc<<2) + 32] = beta * dcptr[4 * (ldc<<2) + 32] + alpha * C_reg[12];
	dcptr[5 * (ldc<<2) + 32] = beta * dcptr[5 * (ldc<<2) + 32] + alpha * C_reg[13];
	dcptr[6 * (ldc<<2) + 32] = beta * dcptr[6 * (ldc<<2) + 32] + alpha * C_reg[14];
	dcptr[7 * (ldc<<2) + 32] = beta * dcptr[7 * (ldc<<2) + 32] + alpha * C_reg[15];
}

__global__ void mysgemm_ilp_v4(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[16] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<5) + threadIdx.y) * lda;
	const float *  dbptr = dB + (blockIdx.x<<6) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<5) + threadIdx.y) * ldc + (blockIdx.x<<6) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 32) {
		float * Bs = &smem[(threadIdx.y<<6) + threadIdx.x];
			
		Bs[0 * (1<<8)] = dbptr[0 * (ldb << 2)];
		Bs[1 * (1<<8)] = dbptr[1 * (ldb << 2)];
		Bs[2 * (1<<8)] = dbptr[2 * (ldb << 2)];
		Bs[3 * (1<<8)] = dbptr[3 * (ldb << 2)];
		Bs[4 * (1<<8)] = dbptr[4 * (ldb << 2)];
		Bs[5 * (1<<8)] = dbptr[5 * (ldb << 2)];
		Bs[6 * (1<<8)] = dbptr[6 * (ldb << 2)];
		Bs[7 * (1<<8)] = dbptr[7 * (ldb << 2)];
		
		Bs[0 * (1<<8) + 32] = dbptr[0 * (ldb << 2) + 32];
		Bs[1 * (1<<8) + 32] = dbptr[1 * (ldb << 2) + 32];
		Bs[2 * (1<<8) + 32] = dbptr[2 * (ldb << 2) + 32];
		Bs[3 * (1<<8) + 32] = dbptr[3 * (ldb << 2) + 32];
		Bs[4 * (1<<8) + 32] = dbptr[4 * (ldb << 2) + 32];
		Bs[5 * (1<<8) + 32] = dbptr[5 * (ldb << 2) + 32];
		Bs[6 * (1<<8) + 32] = dbptr[6 * (ldb << 2) + 32];
		Bs[7 * (1<<8) + 32] = dbptr[7 * (ldb << 2) + 32];
		
		__syncthreads();
		
		Bs = &smem[threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += daptr[0 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[1] += daptr[1 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[2] += daptr[2 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[3] += daptr[3 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[4] += daptr[4 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[5] += daptr[5 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[6] += daptr[6 * (lda<<2) + kk] * Bs[(kk<<6)];
			C_reg[7] += daptr[7 * (lda<<2) + kk] * Bs[(kk<<6)];
		
			C_reg[ 8] += daptr[0 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[ 9] += daptr[1 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[10] += daptr[2 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[11] += daptr[3 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[12] += daptr[4 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[13] += daptr[5 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[14] += daptr[6 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
			C_reg[15] += daptr[7 * (lda<<2) + kk] * Bs[(kk<<6) + 32];
		}
		__syncthreads();				

		daptr += 32;
		dbptr += (ldb << 5);
	}
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[ 0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[ 1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[ 2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[ 3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[ 4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[ 5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[ 6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[ 7 * (ldc<<2)] + alpha * C_reg[7];
	dcptr[0 * (ldc<<2) + 32] = beta * dcptr[0 * (ldc<<2) + 32] + alpha * C_reg[8];
	dcptr[1 * (ldc<<2) + 32] = beta * dcptr[1 * (ldc<<2) + 32] + alpha * C_reg[9];
	dcptr[2 * (ldc<<2) + 32] = beta * dcptr[2 * (ldc<<2) + 32] + alpha * C_reg[10];
	dcptr[3 * (ldc<<2) + 32] = beta * dcptr[3 * (ldc<<2) + 32] + alpha * C_reg[11];
	dcptr[4 * (ldc<<2) + 32] = beta * dcptr[4 * (ldc<<2) + 32] + alpha * C_reg[12];
	dcptr[5 * (ldc<<2) + 32] = beta * dcptr[5 * (ldc<<2) + 32] + alpha * C_reg[13];
	dcptr[6 * (ldc<<2) + 32] = beta * dcptr[6 * (ldc<<2) + 32] + alpha * C_reg[14];
	dcptr[7 * (ldc<<2) + 32] = beta * dcptr[7 * (ldc<<2) + 32] + alpha * C_reg[15];
}

__global__ void mysgemm_ilp_v5(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[2048];
	float C_reg[32] = {0.f};
	
	const float *  daptr = dA + ((blockIdx.y<<6) + threadIdx.y) * lda + threadIdx.x;
	const float *  dbptr = dB + (blockIdx.x<<6) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<6) + threadIdx.y) * ldc + (blockIdx.x<<6) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 16) {
		float * As = &smem[(threadIdx.y<<6) + threadIdx.x];
		float * Bs = &smem[1024 + (threadIdx.y<<6) + threadIdx.x];
		
		Bs[0 * (1<<8)]      = dbptr[0 * (ldb << 2)];
		Bs[1 * (1<<8)]      = dbptr[1 * (ldb << 2)];
		Bs[2 * (1<<8)]      = dbptr[2 * (ldb << 2)];
		Bs[3 * (1<<8)]      = dbptr[3 * (ldb << 2)];
		Bs[0 * (1<<8) + 32] = dbptr[0 * (ldb << 2) + 32];
		Bs[1 * (1<<8) + 32] = dbptr[1 * (ldb << 2) + 32];
		Bs[2 * (1<<8) + 32] = dbptr[2 * (ldb << 2) + 32];
		Bs[3 * (1<<8) + 32] = dbptr[3 * (ldb << 2) + 32];
		__syncthreads();
		
		As = &smem[(threadIdx.y<<4)];
		Bs = &smem[1024 + threadIdx.x];
		
		#pragma unroll
		for(int kk = 0; kk < 16; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<6)];
		
			C_reg[ 8] += As[0 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[ 9] += As[1 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[10] += As[2 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[11] += As[3 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[12] += As[4 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[13] += As[5 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[14] += As[6 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[15] += As[7 * (1<<7) + kk] * Bs[(kk<<6) + 32];
		
			C_reg[16] += As[0 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[17] += As[1 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[18] += As[2 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[19] += As[3 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[20] += As[4 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[21] += As[5 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[22] += As[6 * (1<<7) + kk] * Bs[(kk<<6)];
			C_reg[23] += As[7 * (1<<7) + kk] * Bs[(kk<<6)];
		
			C_reg[24] += As[0 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[25] += As[1 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[26] += As[2 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[27] += As[3 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[28] += As[4 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[29] += As[5 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[30] += As[6 * (1<<7) + kk] * Bs[(kk<<6) + 32];
			C_reg[31] += As[7 * (1<<7) + kk] * Bs[(kk<<6) + 32];
		}
		__syncthreads();
	}	
	
	dcptr[0 * (ldc<<2)] = beta * dcptr[ 0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[1 * (ldc<<2)] = beta * dcptr[ 1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[2 * (ldc<<2)] = beta * dcptr[ 2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[3 * (ldc<<2)] = beta * dcptr[ 3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[4 * (ldc<<2)] = beta * dcptr[ 4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[5 * (ldc<<2)] = beta * dcptr[ 5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[6 * (ldc<<2)] = beta * dcptr[ 6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[7 * (ldc<<2)] = beta * dcptr[ 7 * (ldc<<2)] + alpha * C_reg[7];
	dcptr[0 * (ldc<<2) + 32] = beta * dcptr[0 * (ldc<<2) + 32] + alpha * C_reg[8];
	dcptr[1 * (ldc<<2) + 32] = beta * dcptr[1 * (ldc<<2) + 32] + alpha * C_reg[9];
	dcptr[2 * (ldc<<2) + 32] = beta * dcptr[2 * (ldc<<2) + 32] + alpha * C_reg[10];
	dcptr[3 * (ldc<<2) + 32] = beta * dcptr[3 * (ldc<<2) + 32] + alpha * C_reg[11];
	dcptr[4 * (ldc<<2) + 32] = beta * dcptr[4 * (ldc<<2) + 32] + alpha * C_reg[12];
	dcptr[5 * (ldc<<2) + 32] = beta * dcptr[5 * (ldc<<2) + 32] + alpha * C_reg[13];
	dcptr[6 * (ldc<<2) + 32] = beta * dcptr[6 * (ldc<<2) + 32] + alpha * C_reg[14];
	dcptr[7 * (ldc<<2) + 32] = beta * dcptr[7 * (ldc<<2) + 32] + alpha * C_reg[15];

	dcptr[0 * (ldc<<2)] = beta * dcptr[ 0 * (ldc<<2)] + alpha * C_reg[16];
	dcptr[1 * (ldc<<2)] = beta * dcptr[ 1 * (ldc<<2)] + alpha * C_reg[17];
	dcptr[2 * (ldc<<2)] = beta * dcptr[ 2 * (ldc<<2)] + alpha * C_reg[18];
	dcptr[3 * (ldc<<2)] = beta * dcptr[ 3 * (ldc<<2)] + alpha * C_reg[19];
	dcptr[4 * (ldc<<2)] = beta * dcptr[ 4 * (ldc<<2)] + alpha * C_reg[20];
	dcptr[5 * (ldc<<2)] = beta * dcptr[ 5 * (ldc<<2)] + alpha * C_reg[21];
	dcptr[6 * (ldc<<2)] = beta * dcptr[ 6 * (ldc<<2)] + alpha * C_reg[22];
	dcptr[7 * (ldc<<2)] = beta * dcptr[ 7 * (ldc<<2)] + alpha * C_reg[23];
	dcptr[0 * (ldc<<2) + 32] = beta * dcptr[0 * (ldc<<2) + 32] + alpha * C_reg[24];
	dcptr[1 * (ldc<<2) + 32] = beta * dcptr[1 * (ldc<<2) + 32] + alpha * C_reg[25];
	dcptr[2 * (ldc<<2) + 32] = beta * dcptr[2 * (ldc<<2) + 32] + alpha * C_reg[26];
	dcptr[3 * (ldc<<2) + 32] = beta * dcptr[3 * (ldc<<2) + 32] + alpha * C_reg[27];
	dcptr[4 * (ldc<<2) + 32] = beta * dcptr[4 * (ldc<<2) + 32] + alpha * C_reg[28];
	dcptr[5 * (ldc<<2) + 32] = beta * dcptr[5 * (ldc<<2) + 32] + alpha * C_reg[29];
	dcptr[6 * (ldc<<2) + 32] = beta * dcptr[6 * (ldc<<2) + 32] + alpha * C_reg[30];
	dcptr[7 * (ldc<<2) + 32] = beta * dcptr[7 * (ldc<<2) + 32] + alpha * C_reg[31];
}

__global__ void mysgemm_ilp_v6(const float alpha, const float * __restrict__ dA, const int lda, const float * __restrict__ dB, const int ldb, const float beta, float * __restrict__ dC, const int ldc)
{
	__shared__ float smem[3072];
	float C_reg[16] = {0.f};
	
	const float * daptr = dA + ((blockIdx.y<<6) + threadIdx.y) * lda + threadIdx.x;
	const float * dbptr = dB + (blockIdx.x<<5) + threadIdx.y * ldb + threadIdx.x;
	float *  dcptr = dC + ((blockIdx.y<<6) + threadIdx.y) * ldc + (blockIdx.x<<5) + threadIdx.x;

	for(int ik = 0; ik < lda; ik += 32) {
		float * As = &smem[(threadIdx.y<<5) + threadIdx.x];
		float * Bs = &smem[2048 + (threadIdx.y<<5) + threadIdx.x];
		
		As[0 * (1<<7)] = daptr[0 * (lda << 2)];
		As[1 * (1<<7)] = daptr[1 * (lda << 2)];
		As[2 * (1<<7)] = daptr[2 * (lda << 2)];
		As[3 * (1<<7)] = daptr[3 * (lda << 2)];
		As[4 * (1<<7)] = daptr[4 * (lda << 2)];
		As[5 * (1<<7)] = daptr[5 * (lda << 2)];
		As[6 * (1<<7)] = daptr[6 * (lda << 2)];
		As[7 * (1<<7)] = daptr[7 * (lda << 2)];
		
		As[ 8 * (1<<7)] = daptr[ 8 * (lda << 2)];
		As[ 9 * (1<<7)] = daptr[ 9 * (lda << 2)];
		As[10 * (1<<7)] = daptr[10 * (lda << 2)];
		As[11 * (1<<7)] = daptr[11 * (lda << 2)];
		As[12 * (1<<7)] = daptr[12 * (lda << 2)];
		As[13 * (1<<7)] = daptr[13 * (lda << 2)];
		As[14 * (1<<7)] = daptr[14 * (lda << 2)];
		As[15 * (1<<7)] = daptr[15 * (lda << 2)];

		Bs[0 * (1<<7)] = dbptr[0 * (ldb << 2)];
		Bs[1 * (1<<7)] = dbptr[1 * (ldb << 2)];
		Bs[2 * (1<<7)] = dbptr[2 * (ldb << 2)];
		Bs[3 * (1<<7)] = dbptr[3 * (ldb << 2)];
		Bs[4 * (1<<7)] = dbptr[4 * (ldb << 2)];
		Bs[5 * (1<<7)] = dbptr[5 * (ldb << 2)];
		Bs[6 * (1<<7)] = dbptr[6 * (ldb << 2)];
		Bs[7 * (1<<7)] = dbptr[7 * (ldb << 2)];
	
		__syncthreads();
		
		As = &smem[(threadIdx.y<<5)];
		Bs = &smem[2048 + threadIdx.x];
		#pragma unroll
		for(int kk = 0; kk < 32; kk++) {
			C_reg[0] += As[0 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[1] += As[1 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[2] += As[2 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[3] += As[3 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[4] += As[4 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[5] += As[5 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[6] += As[6 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[7] += As[7 * (1<<7) + kk] * Bs[(kk<<5)];
		
			C_reg[ 8] += As[8 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[ 9] += As[9 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[10] += As[10 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[11] += As[11 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[12] += As[12 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[13] += As[13 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[14] += As[14 * (1<<7) + kk] * Bs[(kk<<5)];
			C_reg[15] += As[15 * (1<<7) + kk] * Bs[(kk<<5)];
		}
		__syncthreads();				

		daptr += 32;
		dbptr += (ldb << 5);
	}
	
	dcptr[ 0 * (ldc<<2)] = beta * dcptr[ 0 * (ldc<<2)] + alpha * C_reg[0];
	dcptr[ 1 * (ldc<<2)] = beta * dcptr[ 1 * (ldc<<2)] + alpha * C_reg[1];
	dcptr[ 2 * (ldc<<2)] = beta * dcptr[ 2 * (ldc<<2)] + alpha * C_reg[2];
	dcptr[ 3 * (ldc<<2)] = beta * dcptr[ 3 * (ldc<<2)] + alpha * C_reg[3];
	dcptr[ 4 * (ldc<<2)] = beta * dcptr[ 4 * (ldc<<2)] + alpha * C_reg[4];
	dcptr[ 5 * (ldc<<2)] = beta * dcptr[ 5 * (ldc<<2)] + alpha * C_reg[5];
	dcptr[ 6 * (ldc<<2)] = beta * dcptr[ 6 * (ldc<<2)] + alpha * C_reg[6];
	dcptr[ 7 * (ldc<<2)] = beta * dcptr[ 7 * (ldc<<2)] + alpha * C_reg[7];
	dcptr[ 8 * (ldc<<2)] = beta * dcptr[ 8 * (ldc<<2)] + alpha * C_reg[8];
	dcptr[ 9 * (ldc<<2)] = beta * dcptr[ 9 * (ldc<<2)] + alpha * C_reg[9];
	dcptr[10 * (ldc<<2)] = beta * dcptr[10 * (ldc<<2)] + alpha * C_reg[10];
	dcptr[11 * (ldc<<2)] = beta * dcptr[11 * (ldc<<2)] + alpha * C_reg[11];
	dcptr[12 * (ldc<<2)] = beta * dcptr[12 * (ldc<<2)] + alpha * C_reg[12];
	dcptr[13 * (ldc<<2)] = beta * dcptr[13 * (ldc<<2)] + alpha * C_reg[13];
	dcptr[14 * (ldc<<2)] = beta * dcptr[14 * (ldc<<2)] + alpha * C_reg[14];
	dcptr[15 * (ldc<<2)] = beta * dcptr[15 * (ldc<<2)] + alpha * C_reg[15];
}

template<size_t BM, size_t BK, size_t BN, size_t TX, size_t TY>
void mygemm_wrapper(const int M, const int K, const int N, const float alpha, const float * A, const int lda, const float * B, const int ldb, const float beta, float * C, const int ldc)
{
	CudaMatrix<BK, BM> wrapperA;
	wrapperA.allocate(M, lda, false, nullptr, const_cast<float*>(A));
	wrapperA.download();
	
	CudaMatrix<BN, BK> wrapperB;
	wrapperB.allocate(K, ldb, false, nullptr, const_cast<float*>(B));
	wrapperB.download();

	CudaMatrix<BN, BM> wrapperC;
	wrapperC.allocate(M, ldc, false, nullptr, C);
	wrapperC.download();

#ifdef VERBOSE
	fprintf(stdout, "INFO: matrix A, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperA.padM, wrapperA.padN);
	fprintf(stdout, "INFO: matrix B, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperB.padM, wrapperB.padN);
	fprintf(stdout, "INFO: matrix C, size = (%dx%d), padding size = (%dx%d)\n", M, K, wrapperC.padM, wrapperC.padN);
#endif

	dim3 grid( wrapperC.padN / BN, wrapperA.padM / BM, 1 );
	dim3 threads( TX, TY, 1 );
	
	TimerGPU timer(0);
	mysgemm_cache_B<BM, BK, BN, TX, TY><<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
//	mysgemm_ilp_v3<<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
//	mysgemm_cache_B_unrolling_v1<<<grid, threads>>>(alpha, wrapperA.d_data, wrapperA.padN, wrapperB.d_data, wrapperB.padN, beta, wrapperC.d_data, wrapperC.padN);
	double gpuTime = timer.read();

//	wrapperA.readback();	
//	for(int i = 0; i < M; i++) {
//		for(int j = 0; j < N; j++) {
//			fprintf(stdout, "%02.2f\t", A[i * N + j]);
//		}
//		fprintf(stdout, "\n");
//	}
//	fflush(stdout);


	fprintf(stdout, "INFO: matrix multiply time = %.2f ms.\n", gpuTime);
#ifdef VERBOSE
	fprintf(stdout, "INFO: performance = %f GFLOPS\n", (2.0 * M * N * K) / (gpuTime / 1000.0 * 1e9));
#endif
	fflush(stdout);
	
	wrapperC.readback();
}

void constantInit(float * data, long int size, float val)
{
	for(long int i = 0; i < size; i++) {
		data[i] = val;
	}
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

#ifdef VERBOSE	
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
	mygemm_wrapper<128, 32, 128, 16, 16>(
		M, K, N, 1.f,
		h_A, K, h_B, N, 0.f, h_C, N);
	
	
//	mygemm_wrapper<128, 32, 64, 32, 8>(
//		M, K, N, 1.f,
//		h_A, K, h_B, N, 0.f, h_C, N);

	
//	double t0 = omp_get_wtime();
	TimerCPU timer(3.07 * 1000);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, h_A, K, h_B, N, 0.0f, h_D, N);
	double cpuTime = timer.read();
//	t0 = omp_get_wtime() - t0;
//	cout << t0 << "\n";
#ifdef VERBOSE
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

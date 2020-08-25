#pragma once

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <hiprand.h>
#include <rocblas.h>


// Driver
#define cudaSuccess hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorNotReady hipErrorNotReady

#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

#define cudaDeviceAttr hipDeviceAttribute_t
#define CUfunction_attribute hipFunction_attribute

#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMaxBlockDimX hipDeviceAttributeMaxBlockDimX
#define cudaDevAttrMaxBlockDimY hipDeviceAttributeMaxBlockDimY
#define cudaDevAttrMaxBlockDimZ hipDeviceAttributeMaxBlockDimZ
#define cudaDevAttrMaxGridDimX hipDeviceAttributeMaxGridDimX
#define cudaDevAttrMaxGridDimY hipDeviceAttributeMaxGridDimY
#define cudaDevAttrMaxGridDimZ hipDeviceAttributeMaxGridDimZ
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrMaxRegistersPerBlock hipDeviceAttributeMaxRegistersPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrTotalConstantMemory hipDeviceAttributeTotalConstantMemory
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#define cudaDevAttrMaxPitch hipDeviceAttributeMaxPitch

#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_NUM_REGS HIP_FUNC_ATTRIBUTE_NUM_REGS

#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventInterprocess hipEventInterprocess
#define cudaEventDefault hipEventDefault

#define NVRTC_SUCCESS HIPRTC_SUCCESS
#define NVRTC_ERROR_COMPILATION HIPRTC_ERROR_COMPILATION

#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaMemGetInfo hipMemGetInfo

#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cuDeviceGet hipDeviceGet
#define cuDeviceGetName hipDeviceGetName
#define cuDeviceTotalMem hipDeviceTotalMem

#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpy3D hipMemcpy3D
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpy3DParms hipMemcpy3DParms
#define make_cudaPitchedPtr make_hipPitchedPtr
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos

#define cuMemsetD8 hipMemsetD8
#define cuMemsetD8Async hipMemsetD8Async
#define cuMemsetD16 hipMemsetD16
#define cuMemsetD16Async hipMemsetD16Async
#define cuMemsetD32 hipMemsetD32
#define cuMemsetD32Async hipMemsetD32Async
#define cuMemcpyDtoH hipMemcpyDtoH
#define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define cuMemcpyHtoD hipMemcpyHtoD
#define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define cuMemcpyDtoD hipMemcpyDtoD
#define cuMemcpyDtoDAsync hipMemcpyDtoDAsync

#define cuModuleLoadData hipModuleLoadData
#define cuModuleUnload hipModuleUnload
#define cuModuleGetFunction hipModuleGetFunction
#define cuFuncGetAttribute hipFuncGetAttribute
#define cuLaunchKernel hipModuleLaunchKernel

#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamWaitEvent hipStreamWaitEvent

#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

#define nvrtcGetErrorString hiprtcGetErrorString
#define nvrtcVersion hiprtcVersion
#define nvrtcCreateProgram hiprtcCreateProgram
#define nvrtcCompileProgram hiprtcCompileProgram
#define nvrtcGetPTXSize hiprtcGetCodeSize
#define nvrtcGetPTX hiprtcGetCode
#define nvrtcGetProgramLogSize hiprtcGetProgramLogSize
#define nvrtcGetProgramLog hiprtcGetProgramLog
#define nvrtcDestroyProgram hiprtcDestroyProgram

#define cudaProfilerStop hipProfilerStop

typedef hipError_t cudaError_t;
typedef hipError_t CUresult;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
typedef hipIpcMemHandle_t cudaIpcMemHandle_t;
typedef void *CUdeviceptr;
typedef hipDevice_t CUdevice;
typedef hipModule_t CUmodule;
typedef hipFunction_t CUfunction;

typedef hiprtcResult nvrtcResult;
typedef hiprtcProgram nvrtcProgram;


// Rand
#define CURAND_STATUS_SUCCESS HIPRAND_STATUS_SUCCESS

#define CURAND_RNG_TEST HIPRAND_RNG_TEST
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define CURAND_RNG_PSEUDO_XORWOW HIPRAND_RNG_PSEUDO_XORWOW
#define CURAND_RNG_PSEUDO_MRG32K3A HIPRAND_RNG_PSEUDO_MRG32K3A
#define CURAND_RNG_PSEUDO_MTGP32 HIPRAND_RNG_PSEUDO_MTGP32
#define CURAND_RNG_PSEUDO_MT19937 HIPRAND_RNG_PSEUDO_MT19937
#define CURAND_RNG_PSEUDO_PHILOX4_32_10 HIPRAND_RNG_PSEUDO_PHILOX4_32_10
#define CURAND_RNG_QUASI_DEFAULT HIPRAND_RNG_QUASI_DEFAULT
#define CURAND_RNG_QUASI_SOBOL32 HIPRAND_RNG_QUASI_SOBOL32
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
#define CURAND_RNG_QUASI_SOBOL64 HIPRAND_RNG_QUASI_SOBOL64
#define CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64

#define curandGetVersion hiprandGetVersion
#define curandCreateGenerator hiprandCreateGenerator
#define curandDestroyGenerator hiprandDestroyGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandSetGeneratorOffset hiprandSetGeneratorOffset
#define curandGenerate hiprandGenerate
#define curandGenerateUniform hiprandGenerateUniform
#define curandGenerateUniformDouble hiprandGenerateUniformDouble
#define curandGenerateNormal hiprandGenerateNormal
#define curandGenerateNormalDouble hiprandGenerateNormalDouble

typedef hiprandStatus_t curandStatus_t;
typedef hiprandGenerator_t curandGenerator_t;
typedef hiprandRngType_t curandRngType_t;

inline static const char *curandGetErrorString(curandStatus_t code)
{
	switch (code)
	{
		case HIPRAND_STATUS_SUCCESS:                   return TO_STRING(HIPRAND_STATUS_SUCCESS);
		case HIPRAND_STATUS_VERSION_MISMATCH:          return TO_STRING(HIPRAND_STATUS_VERSION_MISMATCH);
		case HIPRAND_STATUS_NOT_INITIALIZED:           return TO_STRING(HIPRAND_STATUS_NOT_INITIALIZED);
		case HIPRAND_STATUS_ALLOCATION_FAILED:         return TO_STRING(HIPRAND_STATUS_ALLOCATION_FAILED);
		case HIPRAND_STATUS_TYPE_ERROR:                return TO_STRING(HIPRAND_STATUS_TYPE_ERROR);
		case HIPRAND_STATUS_OUT_OF_RANGE:              return TO_STRING(HIPRAND_STATUS_OUT_OF_RANGE);
		case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:       return TO_STRING(HIPRAND_STATUS_LENGTH_NOT_MULTIPLE);
		case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED: return TO_STRING(HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED);
		case HIPRAND_STATUS_LAUNCH_FAILURE:            return TO_STRING(HIPRAND_STATUS_LAUNCH_FAILURE);
		case HIPRAND_STATUS_PREEXISTING_FAILURE:       return TO_STRING(HIPRAND_STATUS_PREEXISTING_FAILURE);
		case HIPRAND_STATUS_INITIALIZATION_FAILED:     return TO_STRING(HIPRAND_STATUS_INITIALIZATION_FAILED);
		case HIPRAND_STATUS_ARCH_MISMATCH:             return TO_STRING(HIPRAND_STATUS_ARCH_MISMATCH);
		case HIPRAND_STATUS_INTERNAL_ERROR:            return TO_STRING(HIPRAND_STATUS_INTERNAL_ERROR);
		default:                                       assert(false); return "UNKNOWN_ERROR";
	}
}


// Blas
#define CUBLAS_STATUS_SUCCESS rocblas_status_success

#define CUDA_R_32F rocblas_datatype_f32_r
#define CUDA_R_16F rocblas_datatype_f16_r

#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose

#define CUBLAS_GEMM_DEFAULT rocblas_gemm_algo_standard
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP rocblas_gemm_algo_standard

#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasSasum rocblas_sasum
#define cublasSnrm2 rocblas_snrm2
#define cublasSdot rocblas_sdot

typedef rocblas_status cublasStatus_t;
typedef rocblas_handle cublasHandle_t;
typedef rocblas_datatype cudaDataType;
typedef rocblas_operation cublasOperation_t;
typedef rocblas_gemm_algo cublasGemmAlgo_t;

inline static cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version)
{
	(void)handle;

	*version = ROCBLAS_VERSION_MAJOR * 10000 + ROCBLAS_VERSION_MINOR * 100 + ROCBLAS_VERSION_PATCH;
	return CUBLAS_STATUS_SUCCESS;
}

inline static const char *cublasGetErrorString(cublasStatus_t code)
{
	switch (code)
	{
		case rocblas_status_success:             return TO_STRING(rocblas_status_success);
		case rocblas_status_invalid_handle:      return TO_STRING(rocblas_status_invalid_handle);
		case rocblas_status_not_implemented:     return TO_STRING(rocblas_status_not_implemented);
		case rocblas_status_invalid_pointer:     return TO_STRING(rocblas_status_invalid_pointer);
		case rocblas_status_invalid_size:        return TO_STRING(rocblas_status_invalid_size);
		case rocblas_status_memory_error:        return TO_STRING(rocblas_status_memory_error);
		case rocblas_status_internal_error:      return TO_STRING(rocblas_status_internal_error);
		case rocblas_status_perf_degraded:       return TO_STRING(rocblas_status_perf_degraded);
		case rocblas_status_size_query_mismatch: return TO_STRING(rocblas_status_size_query_mismatch);
		case rocblas_status_size_increased:      return TO_STRING(rocblas_status_size_increased);
		case rocblas_status_size_unchanged:      return TO_STRING(rocblas_status_size_unchanged);
		default:                                 assert(false); return "UNKNOWN_ERROR";
	}
}

#pragma push_macro("rocblas_gemm_ex")
#undef rocblas_gemm_ex
inline static cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
										  int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
										  int lda, const void *B, cudaDataType Btype, int ldb, const void *beta,
										  void *C, cudaDataType Ctype, int ldc, cudaDataType computeType,
										  cublasGemmAlgo_t algo)
{
	return rocblas_gemm_ex(
		handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, C, Ctype, ldc,
		computeType, algo, 0, 0
	);
}
#pragma pop_macro("rocblas_gemm_ex")

#pragma push_macro("rocblas_gemm_strided_batched_ex")
#undef rocblas_gemm_strided_batched_ex
inline static cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
														cublasOperation_t transb, int m, int n, int k,
														const void *alpha, const void *A, cudaDataType Atype, int lda,
														long long int strideA, const void *B, cudaDataType Btype,
														int ldb, long long int strideB, const void *beta, void *C,
														cudaDataType Ctype, int ldc, long long int strideC,
														int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo)
{
	return rocblas_gemm_strided_batched_ex(
		handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc,
		strideC, C, Ctype, ldc, strideC, batchCount, computeType, algo, 0, 0
	);
}
#pragma pop_macro("rocblas_gemm_strided_batched_ex")

/*
 * This is an OpenCL implementation of the reduction kernels described my Mark Harris in
 *
 *     http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 * We run our examples for 2^{22} floats 10000 times with
 *
 *   ./reduction  4194304 10000
 *
 * or 2^{25} floats 100 times
 *
 *   ./reduction 33554432 100
 *
 * OpenCL implementation by Lucas Wilcox
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timing.h"
#include "cl-helper.h"

#define LDIM 256

#define STR(a) #a
#define STRINGIFY(a) STR(a)

int main (int argc, char *argv[])
{
  float *a, *a_reduced;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s N nloops\n", argv[0]);
    abort();
  }

  const cl_long N = (cl_long) atol(argv[1]);
  const int nloops = atoi(argv[2]);

  cl_long Ngroups = (N + LDIM  - 1)/LDIM;

  // half the number of groups for reduction4
  // Ngroups = (Ngroups + 2  - 1)/2;

  // quarter the number of groups for reduction5
  // Ngroups = (Ngroups + 4  - 1)/4;

  Ngroups = (Ngroups + 8  - 1)/8;

  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);

  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("reduction.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "reduction7",
      "-DLDIM=" STRINGIFY(LDIM));
  free(knl_text);

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------
  posix_memalign((void**)&a, 32, N*sizeof(float));
  if (!a) { fprintf(stderr, "alloc a"); abort(); }
  posix_memalign((void**)&a_reduced, 32, Ngroups*sizeof(float));
  if (!a_reduced) { fprintf(stderr, "alloc a_reduced"); abort(); }

  srand48(8);
  for(cl_long n = 0; n < N; ++n)
    //a[n] = (float)drand48();
    a[n] = n;

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N*sizeof(float),
      0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_a_reduced = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      N*sizeof(float), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        N*sizeof(float), a,
        0, NULL, NULL));

  timestamp_type tic, toc;
  double elapsed;

  // --------------------------------------------------------------------------
  // run reduction_simple on device
  // --------------------------------------------------------------------------

  printf("Simple Reduction\n");
  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&tic);
  for(int loop = 0; loop < nloops; ++loop)
  {
    SET_3_KERNEL_ARGS(knl, N, buf_a, buf_a_reduced);

    size_t local_size[] = { LDIM };
    size_t global_size[] = { Ngroups*LDIM };

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 1, NULL,
          global_size, local_size, 0, NULL, NULL));
  }
  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&toc);

  elapsed = timestamp_diff_in_seconds(tic,toc)/nloops;
  printf("%f s\n", elapsed);
  printf("%f GB/s\n", N*sizeof(float)/1e9/elapsed);

  // --------------------------------------------------------------------------
  // transfer back & check
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clEnqueueReadBuffer(
        queue, buf_a_reduced, /*blocking*/ CL_TRUE, /*offset*/ 0,
        Ngroups*sizeof(float), a_reduced, 0, NULL, NULL));

  float sum_cpu = 0.0f;
  for(cl_long n = 0; n < N; ++n)
    sum_cpu += a[n];

  printf("Sum CPU: %e\n", sum_cpu);

  float sum_gpu = 0.0f;
  for(cl_long n = 0; n < Ngroups; ++n)
    sum_gpu += a_reduced[n];

  printf("Sum GPU: %e\n", sum_gpu);

  printf("Relative Error: %e\n", fabsf(sum_cpu-sum_gpu)/sum_gpu);

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clReleaseMemObject(buf_a));
  CALL_CL_SAFE(clReleaseMemObject(buf_a_reduced));
  CALL_CL_SAFE(clReleaseKernel(knl));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));

  free(a);
  free(a_reduced);

  return 0;
}

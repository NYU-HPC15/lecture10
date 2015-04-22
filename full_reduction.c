/*
 * This is an OpenCL implementation of the reduction kernels described my Mark Harris in
 *
 *     http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 * We run our examples for 2^{22} doubles 10000 times with
 *
 *   ./full_reduction  4194304 10000
 *
 * or 2^{25} floats 100 times
 *
 *   ./full_reduction 33554432 100
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
  double *a, *a_reduced;

  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s N nloops\n", argv[0]);
    abort();
  }

  const cl_long N = (cl_long) atol(argv[1]);
  const int nloops = atoi(argv[2]);

  cl_long Ngroups = (N + LDIM  - 1)/LDIM;
  Ngroups = (Ngroups + 8  - 1)/8;

  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);

  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("full_reduction.cl");
  cl_kernel knl = kernel_from_string(ctx, knl_text, "reduction",
      "-DLDIM=" STRINGIFY(LDIM));
  free(knl_text);

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------
  posix_memalign((void**)&a, 32, N*sizeof(double));
  if (!a) { fprintf(stderr, "alloc a"); abort(); }
  posix_memalign((void**)&a_reduced, 32, Ngroups*sizeof(double));
  if (!a_reduced) { fprintf(stderr, "alloc a_reduced"); abort(); }

  srand48(8);
  for(cl_long n = 0; n < N; ++n)
    a[n] = (double)drand48();
    // a[n] = n;

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N*sizeof(double),
      0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem buf_a_reduced[2];
  buf_a_reduced[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      Ngroups*sizeof(double), 0, &status);
  buf_a_reduced[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      Ngroups*sizeof(double), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // transfer to device
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        N*sizeof(double), a,
        0, NULL, NULL));

  timestamp_type tic, toc;
  double elapsed;

  // --------------------------------------------------------------------------
  // run reduction_simple on device
  // --------------------------------------------------------------------------

  printf("Simple Reduction\n");
  double sum_gpu = 0.0;
  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&tic);
  for(int loop = 0; loop < nloops; ++loop)
  {
    int r = 0;
    size_t Ngroups_loop = Ngroups;
    SET_3_KERNEL_ARGS(knl, N, buf_a, buf_a_reduced[r]);

    size_t local_size[] = { LDIM };
    size_t global_size[] = { Ngroups_loop*LDIM };

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 1, NULL,
          global_size, local_size, 0, NULL, NULL));

    while(Ngroups_loop > 1)
    {
      cl_long N_reduce = Ngroups_loop;
      Ngroups_loop = (N_reduce + LDIM  - 1)/LDIM;
      Ngroups_loop = (Ngroups_loop + 8  - 1)/8;

      size_t local_size[] = { LDIM };
      size_t global_size[] = { Ngroups_loop*LDIM };

      SET_3_KERNEL_ARGS(knl, N_reduce, buf_a_reduced[r], buf_a_reduced[(r+1)%2]);

      CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl, 1, NULL,
            global_size, local_size, 0, NULL, NULL));

      r = (r+1)%2;
    }

    CALL_CL_SAFE(clEnqueueReadBuffer(
          queue, buf_a_reduced[r], /*blocking*/ CL_TRUE, /*offset*/ 0,
          Ngroups_loop*sizeof(double), a_reduced, 0, NULL, NULL));

    sum_gpu = 0.0;
    for(cl_long n = 0; n < Ngroups_loop; ++n)
      sum_gpu += a_reduced[n];
  }
  CALL_CL_SAFE(clFinish(queue));
  get_timestamp(&toc);

  elapsed = timestamp_diff_in_seconds(tic,toc)/nloops;
  printf("%f s\n", elapsed);
  printf("%f GB/s\n", N*sizeof(double)/1e9/elapsed);

  double sum_cpu = 0.0;
  for(cl_long n = 0; n < N; ++n)
    sum_cpu += a[n];

  printf("Sum CPU: %e\n", sum_cpu);

  printf("Sum GPU: %e\n", sum_gpu);

  printf("Relative Error: %e\n", fabs(sum_cpu-sum_gpu)/sum_gpu);

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_SAFE(clReleaseMemObject(buf_a));
  CALL_CL_SAFE(clReleaseMemObject(buf_a_reduced[0]));
  CALL_CL_SAFE(clReleaseMemObject(buf_a_reduced[1]));
  CALL_CL_SAFE(clReleaseKernel(knl));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));

  free(a);
  free(a_reduced);

  return 0;
}

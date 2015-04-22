/*
 * This is an OpenCL implementation of the reduction kernels described my Mark Harris in
 *
 *     http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */

kernel void reduction(long N,
                      global const double * restrict a,
                      global       double * restrict a_reduced)
{
  local double a_local[LDIM];

  double sum = 0.0;

  size_t idx = get_global_id(0);
  if(idx < N)
    sum = a[idx];

  idx += get_global_size(0);
  if(idx < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  idx += get_global_size(0);
  if(idx  < N)
    sum += a[idx];

  a_local[get_local_id(0)] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  /*
   * Assume is a power of 2 such that LDIM <= 1024
   */
  if(LDIM >= 1024)
  {
    if(get_local_id(0) < 512)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 512];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 512)
  {
    if(get_local_id(0) < 256)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 256];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 256)
  {
    if(get_local_id(0) < 128)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 128];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 128)
  {
    if(get_local_id(0) < 64)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 64];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 64)
  {
    if(get_local_id(0) < 32)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 32];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 32)
  {
    if(get_local_id(0) < 16)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 16];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 16)
  {
    if(get_local_id(0) < 8)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 8];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 8)
  {
    if(get_local_id(0) < 4)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 4];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 4)
  {
    if(get_local_id(0) < 2)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 2];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(LDIM >= 2)
  {
    if(get_local_id(0) < 1)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + 1];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

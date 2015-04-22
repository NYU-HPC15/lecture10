/*
 * This is an OpenCL implementation of the reduction kernels described my Mark Harris in
 *
 *     http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */

kernel void reduction1(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  if(get_global_id(0) < N)
    a_local[get_local_id(0)] = a[get_global_id(0)];
  else
    a_local[get_local_id(0)] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(size_t s=1; s < LDIM; s*=2)
  {
    if(get_local_id(0) % (2*s) == 0)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + s];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

kernel void reduction2(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  if(get_global_id(0) < N)
    a_local[get_local_id(0)] = a[get_global_id(0)];
  else
    a_local[get_local_id(0)] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(size_t s=1; s < LDIM; s*=2)
  {
    size_t index = 2 * s * get_local_id(0);

    if(index < LDIM)
      a_local[index] += a_local[index + s];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

kernel void reduction3(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  if(get_global_id(0) < N)
    a_local[get_local_id(0)] = a[get_global_id(0)];
  else
    a_local[get_local_id(0)] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(size_t s=LDIM/2; s>0; s>>=1) // Right shift by 1-bit is division by 2
  {
    if(get_local_id(0) < s)
      a_local[get_local_id(0)] += a_local[get_local_id(0)+ s];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

kernel void reduction4(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  float sum = 0.0f;

  if(get_global_id(0) < N)
    sum = a[get_global_id(0)];

  if(2*get_global_id(0) < N)
    sum += a[get_global_size(0)+get_global_id(0)];

  a_local[get_local_id(0)] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(size_t s=LDIM/2; s>0; s>>=1) // Right shift by 1-bit is division by 2
  {
    if(get_local_id(0) < s)
      a_local[get_local_id(0)] += a_local[get_local_id(0)+ s];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

kernel void reduction5(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  float sum = 0.0f;

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

  a_local[get_local_id(0)] = sum;

  barrier(CLK_LOCAL_MEM_FENCE);

  for(size_t s=LDIM/2; s>0; s>>=1) // Right shift by 1-bit is division by 2
  {
    if(get_local_id(0) < s)
      a_local[get_local_id(0)] += a_local[get_local_id(0) + s];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(get_local_id(0) == 0)
    a_reduced[get_group_id(0)] = a_local[0];
}

kernel void reduction6(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  float sum = 0.0f;

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

kernel void reduction7(long N,
                       global const float * restrict a,
                       global       float * restrict a_reduced)
{
  local float a_local[LDIM];

  float sum = 0.0f;

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

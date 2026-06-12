#include <stdio.h>

// TODO: Implement kernel
__global__ void vectorScale(float *data, float scale, int n) {
  // Your code here
  // Multiply each element by scale
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    data[i] = data[i] * scale;
  }
}
int main() {
  // Test
  int n = 1000000;
  float scale = 2.5f;
  float *h_data = (float *)malloc(n * sizeof(float));
  float *d_data;
  cudaMalloc(&d_data, n * sizeof(float));
  // TODO: Initialize data
  printf("Initializing data...\n");
  for (int i = 0; i < n; i++) {
    h_data[i] = i;
  }
  // TODO: Copy to device
  printf("Copying to device...\n");
  cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
  // TODO: Launch kernel
  printf("Launching kernel...\n");
  vectorScale<<<7814, 128>>>(d_data, scale, n);
  // TODO: Copy back and verify
  printf("Copying back and verifying...\n");
  cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("First element: %f\n", h_data[0]);
  printf("Last element: %f\n", h_data[n - 1]);
  return 0;
}
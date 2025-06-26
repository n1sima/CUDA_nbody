#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "tipsy.h"
#include <vector>
#include <iostream>

#define BLOCK_SIZE 256
#define EPS2 1e-6f
#define p BLOCK_SIZE  // Tile size (equal to block size)

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    float s = bj.w * invDistCube;
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel) {
    extern __shared__ float4 shPosition[];
    for (int i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

__global__ void calculate_forces(void *devX, void *devA, int N) {
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= N) return;  // Guard out-of-bounds

    float4 myPosition = globalX[gtid];
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int i = 0, tile = 0; i < N; i += p, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < N)
            shPosition[threadIdx.x] = globalX[idx];
        else
            shPosition[threadIdx.x] = make_float4(0,0,0,0); // zero padding
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }

    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}

int main() {
    std::vector<float4> tipsyPositions;
    std::vector<float4> tipsyVelocities;  // (optional, not used here)
    std::vector<int> tipsyIDs;
    int NTotal, NDark, NStar, NSph;

    // Read your TIPSY file (replace "low.bin" with your filename)
    read_tipsy_file(tipsyPositions, tipsyVelocities, tipsyIDs, "low.bin", NTotal, NDark, NStar, NSph);

    if (NTotal == 0) {
        std::cerr << "Error: No bodies read from the TIPSY file." << std::endl;
        return -1;
    }

    printf("Read %d bodies from TIPSY file\n", NTotal);

    // Allocate host memory
    float4* h_X = (float4*)malloc(NTotal * sizeof(float4));
    float4* h_A = (float4*)malloc(NTotal * sizeof(float4));

    // Copy TIPSY positions to host array h_X
    for (int i = 0; i < NTotal; i++) {
        h_X[i] = tipsyPositions[i];
    }

    // Allocate device memory
    float4 *d_X, *d_A;
    cudaMalloc(&d_X, NTotal * sizeof(float4));
    cudaMalloc(&d_A, NTotal * sizeof(float4));

    // Copy positions to device
    cudaMemcpy(d_X, h_X, NTotal * sizeof(float4), cudaMemcpyHostToDevice);

    // Setup kernel launch parameters
    dim3 block(BLOCK_SIZE);
    dim3 grid((NTotal + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float4);

    // Launch kernel with NTotal as argument
    calculate_forces<<<grid, block, sharedMemSize>>>((void *)d_X, (void *)d_A, NTotal);

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_A, d_A, NTotal * sizeof(float4), cudaMemcpyDeviceToHost);

    // Print first 5 acceleration results
    for (int i = 0; i < 5 && i < NTotal; i++) {
        printf("a[%d] = (%f, %f, %f)\n", i, h_A[i].x, h_A[i].y, h_A[i].z);
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_A);
    free(h_X);
    free(h_A);

    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024        // Number of bodies
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

__global__ void calculate_forces(void *devX, void *devA) {
    extern __shared__ float4 shPosition[];
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    float3 acc = {0.0f, 0.0f, 0.0f};

    myPosition = globalX[gtid];
    for (int i = 0, tile = 0; i < N; i += p, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }

    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}

int main() {
    float4 *h_X = (float4 *)malloc(N * sizeof(float4));
    float4 *h_A = (float4 *)malloc(N * sizeof(float4));

    // Initialize input data (positions and masses)
    for (int i = 0; i < N; i++) {
        h_X[i] = make_float4(i * 1e-3f, i * 1e-3f, i * 1e-3f, 1.0f); // Example
    }

    float4 *d_X, *d_A;
    cudaMalloc(&d_X, N * sizeof(float4));
    cudaMalloc(&d_A, N * sizeof(float4));

    cudaMemcpy(d_X, h_X, N * sizeof(float4), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE);
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float4);

    calculate_forces<<<grid, block, sharedMemSize>>>((void *)d_X, (void *)d_A);

    cudaMemcpy(h_A, d_A, N * sizeof(float4), cudaMemcpyDeviceToHost);

    // Print first 5 results
    for (int i = 0; i < 5; i++) {
        printf("a[%d] = (%f, %f, %f)\n", i, h_A[i].x, h_A[i].y, h_A[i].z);
    }

    cudaFree(d_X);
    cudaFree(d_A);
    free(h_X);
    free(h_A);
    return 0;
}

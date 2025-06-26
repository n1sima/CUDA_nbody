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

__global__ void computeForces(const float4* positions, float3* accelerations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 myPos = positions[i];
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < N; ++j) {
        acc = bodyBodyInteraction(myPos, positions[j], acc);
    }

    accelerations[i] = acc;
}

int main() {
    // Load position data from binary file
    std::vector<float4> h_positions(N);
    std::ifstream inFile("positions.bin", std::ios::binary);
    if (!inFile) {
        std::cerr << "Failed to open positions.bin\n";
        return 1;
    }
    inFile.read(reinterpret_cast<char*>(h_positions.data()), N * sizeof(float4));
    inFile.close();

    // Allocate memory on device
    float4* d_positions;
    float3* d_accelerations;
    cudaMalloc(&d_positions, N * sizeof(float4));
    cudaMalloc(&d_accelerations, N * sizeof(float3));

    // Copy data to device
    cudaMemcpy(d_positions, h_positions.data(), N * sizeof(float4), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeForces<<<numBlocks, BLOCK_SIZE>>>(d_positions, d_accelerations);
    cudaDeviceSynchronize();

    // Copy results back (optional)
    std::vector<float3> h_accelerations(N);
    cudaMemcpy(h_accelerations.data(), d_accelerations, N * sizeof(float3), cudaMemcpyDeviceToHost);

    // Print a few results
    for (int i = 0; i < 5; ++i) {
        std::cout << "Accel[" << i << "] = ("
                  << h_accelerations[i].x << ", "
                  << h_accelerations[i].y << ", "
                  << h_accelerations[i].z << ")\n";
    }

    cudaFree(d_positions);
    cudaFree(d_accelerations);
    return 0;
}

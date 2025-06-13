#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define BLOCK_SIZE 256
#define EPS2 1e-6f

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

__device__ float3 tile_calculation(float4 myPos, float3 acc, float4* shPosition) {
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        acc = bodyBodyInteraction(myPos, shPosition[j], acc);
    }
    return acc;
}

__global__ void computeForcesTiled(const float4* positions, float3* accelerations, int numParticles) {
    extern __shared__ float4 shPosition[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 myPos;
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    if (tid < numParticles) {
        myPos = positions[tid];

        for (int tile = 0; tile < gridDim.x; ++tile) {
            int idx = tile * blockDim.x + threadIdx.x;

            if (idx < numParticles)
                shPosition[threadIdx.x] = positions[idx];
            else
                shPosition[threadIdx.x] = make_float4(0, 0, 0, 0); // avoid garbage

            __syncthreads();

            acc = tile_calculation(myPos, acc, shPosition);

            __syncthreads();
        }

        accelerations[tid] = acc;
    }
}

__global__ void integrate(float4* positions, float3* velocities, float3* accelerations, float deltaTime, int numParticles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numParticles) {
        velocities[tid].x += accelerations[tid].x * deltaTime;
        velocities[tid].y += accelerations[tid].y * deltaTime;
        velocities[tid].z += accelerations[tid].z * deltaTime;

        positions[tid].x += velocities[tid].x * deltaTime;
        positions[tid].y += velocities[tid].y * deltaTime;
        positions[tid].z += velocities[tid].z * deltaTime;
    }

}

int main() {
    // Load position data from binary file
    std::vector<float4> h_positions(N);
    std::ifstream inFile("positions.bin", std::ios::binary);

    float3* d_velocities;
    cudaMalloc(&d_velocities, N * sizeof(float3));
    cudaMemset(d_velocities, 0, N * sizeof(float3)); // optional: start with zero velocity

    float4* d_positions[2];
    cudaMalloc(&d_positions[0], N * sizeof(float4));
    cudaMalloc(&d_positions[1], N * sizeof(float4));

    int memRead = 0;
    int memWrite = 1;






    if (!inFile) {
        std::cerr << "Failed to open positions.bin\n";
        return 1;
    }
    inFile.read(reinterpret_cast<char*>(h_positions.data()), N * sizeof(float4));
    inFile.close();

    // Allocate memory on device
    float3* d_accelerations;
    cudaMalloc(&d_accelerations, N * sizeof(float3));

    // Copy data to device
    cudaMemcpy(d_positions[memRead], h_positions.data(), N * sizeof(float4), cudaMemcpyHostToDevice);

    float deltaTime = 0.1f; 



    // Launch kernel
    int numsteps = 5; // Number of simulation steps
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float4);

    cudaMemcpy(d_positions[memWrite], d_positions[memRead], N * sizeof(float4), cudaMemcpyDeviceToDevice);


    for (int step = 0; step < numsteps; ++step) {
        // Swap memory pointers
        computeForcesTiled<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(d_positions[memRead], d_accelerations, N);

        integrate<<<numBlocks, BLOCK_SIZE>>>(
        d_positions[memWrite], d_velocities, d_accelerations, deltaTime, N);

        cudaDeviceSynchronize();
        std::swap(memRead, memWrite);

    }

    
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

    std::swap(memRead, memWrite);


    cudaFree(d_positions[0]);
    cudaFree(d_positions[1]);  
    cudaFree(d_velocities);
    cudaFree(d_accelerations);


    return 0;
}

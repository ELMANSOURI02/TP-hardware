
#include <cstdlib>
#include <ctime>
#include <iostream>

void MatrixInit(float *M, int n, int p) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = (float)rand() / RAND_MAX * 2 - 1;
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            std::cout << M[i * p + j] << " ";
        }
        std::cout << std::endl;
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < p) {
        Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;

        for (int k = 0; k < n; ++k) {
            sum += M1[row * n + k] * M2[k * n + col];
        }

        Mout[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_columns>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int p = std::atoi(argv[2]);

    // Allocate memory for matrices on CPU
    float *M1 = new float[n * p];
    float *M2 = new float[n * p];
    float *Mout = new float[n * p];
    float *Mout2 = new float[n * p];

    // Initialize matrices
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Perform matrix addition on CPU
    MatrixAdd(M1, M2, Mout, n, p);;
    MatrixMult(M1, M2, Mout2, n);

    // Allocate memory for matrices on GPU
    float *d_M1, *d_M2, *d_Mout, *d_Mout2;
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout2, n * p * sizeof(float));


    // Copy matrices from CPU to GPU
    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(n);
    dim3 gridDim(p);

    // Perform matrix addition on GPU
    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();
    

    // multiplication de matrice
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout2, n);
    cudaDeviceSynchronize();

    // Copy result matrix from GPU to CPU
    cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Mout2, d_Mout2, n * p * sizeof(float), cudaMemcpyDeviceToHost);


    // Print the result matrix
    printf("add\n");
    MatrixPrint(Mout, n, p);
    printf("multiplication\n");
    MatrixPrint(Mout2, n, p);

    // Free memory on GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    cudaFree(d_Mout2);
    // Cleanup
    delete[] M1;
    delete[] M2;
    delete[] Mout;
    delete[] Mout2;

    return 0;
}
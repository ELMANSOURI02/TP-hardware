#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2 - 1; 
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%0.2f ", M[i * p + j]);
        }
        printf("\n");
    }
    printf("\n");
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

// Fonction pour multiplier deux matrices NxN sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}
// Fonction pour multiplier deux matrices NxN sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = sum;
    }
}



// Fonction principale
int main() {
    // Dimensions des matrices
    int n = 3, p = 3; // Matrices carrées NxN
    size_t size = n * p * sizeof(float);

    // Allocation mémoire pour CPU
    float *M1 = (float *)malloc(size);
    float *M2 = (float *)malloc(size);
    float *Mout_cpu_add = (float *)malloc(size); // Résultat CPU pour addition
    float *Mout_cpu_mult = (float *)malloc(size); // Résultat CPU pour multiplication
    float *Mout_gpu_add = (float *)malloc(size); // Résultat GPU pour addition
    float *Mout_gpu_mult = (float *)malloc(size); // Résultat GPU pour multiplication

    // Allocation mémoire pour GPU
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc(&d_M1, size);
    cudaMalloc(&d_M2, size);
    cudaMalloc(&d_Mout, size);

    // Initialisation des matrices
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    printf("Matrix 1:\n");
    MatrixPrint(M1, n, p);

    printf("Matrix 2:\n");
    MatrixPrint(M2, n, p);

    // --- PARTIE ADDITION ---

    // Addition sur CPU
    MatrixAdd(M1, M2, Mout_cpu_add, n, p);
    printf("CPU Matrix Addition Result:\n");
    MatrixPrint(Mout_cpu_add, n, p);

    // Copie des matrices sur GPU
    cudaMemcpy(d_M1, M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, size, cudaMemcpyHostToDevice);

    // Configurer les dimensions de la grille et des blocs
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancer le kernel CUDA pour l'addition
    cudaMatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, p);
    cudaDeviceSynchronize();

    // Copier le résultat vers le CPU
    cudaMemcpy(Mout_gpu_add, d_Mout, size, cudaMemcpyDeviceToHost);

    printf("GPU Matrix Addition Result:\n");
    MatrixPrint(Mout_gpu_add, n, p);

    // --- PARTIE MULTIPLICATION ---

    // Multiplication sur CPU
    MatrixMult(M1, M2, Mout_cpu_mult, n);
    printf("CPU Matrix Multiplication Result:\n");
    MatrixPrint(Mout_cpu_mult, n, n);

    // Lancer le kernel CUDA pour la multiplication
    cudaMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n);
    cudaDeviceSynchronize();

    // Copier le résultat vers le CPU
    cudaMemcpy(Mout_gpu_mult, d_Mout, size, cudaMemcpyDeviceToHost);

    printf("GPU Matrix Multiplication Result:\n");
    MatrixPrint(Mout_gpu_mult, n, n);

    // Libérer la mémoire
    free(M1); free(M2); free(Mout_cpu_add); free(Mout_cpu_mult);
    free(Mout_gpu_add); free(Mout_gpu_mult);
    cudaFree(d_M1); cudaFree(d_M2); cudaFree(d_Mout);

    return 0;
}
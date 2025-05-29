#include <stdio.h>

__declspec(dllexport)
void matrix_multiply(int N, int M, int P, int* A, int* B, int* C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// Tile the matrix in 32*32 and then proceed.

// In tiling bring in our 0 approach - embedding layer 0 to be forced to 0 

// Reduce the training compute - or fine tuning compute??
	// take  a pretrained embedding - preferabbly MOE, during training when a token is presented to model to train 
	// the model retrieves already trained embedding zeroed dimensions are clipped off (not considered for matmul) 
	// train only on active dimensions and their weight
	// Improvise - dynamically detect if the compute for that vectors need to be continued or just 0 it and deactivate it.
	// add the reusablity approach of matrices insied deepseek transfomer s
// Study deep seek papers
// 
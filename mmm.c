/* File:     mmm.c
 * Author: Thaison Nghiem
 * Class AUCSC 450
 * 
 * Purpose:  Implement a Matrix-Matrix Multiplication using MPI to 
 *           multiply 𝑚 × 𝑘 matrix by 𝑘 × 𝑛 matrix, the (𝑘)s (number 
 *           of columns in the first matrix (𝐴) and the number of rows 
 *           in the second matrix (𝐵) must be the same, and the result 
 *           (𝐶) must be of size 𝑚 × 𝑛 matrix.
 *
 * Compile:  mpicc -g -Wall -o mmm mmm.c
 * Run:      mpiexec -n <number of processes> ./mmm m k n
 *
 * Input:    m: number of rows in Matrix A, number of rows in Matrix C
 *           k: number of collumns in Matrix A, number of rows in Matrix B
 *           n: number of collumns in Matrix B, number of collumns in Matrix C
 * Output:   A matrix
 *           B matrix
 *           C matrix – computed in parallel with runtime
 *           C matrix – computed sequentially with runtime
 * 
 * Errors:   If an error is detected (m, k or n not evenly
 *           divisible by the number of processes), the
 *           program prints a message and all processes quit.
 *
 * Notes:
 *    1. Number of processes should evenly divide both m and n
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void parMatMul(float* A, float* B, float* C, int m, int k, int n, int size, MPI_Comm comm);
void seqMatMul(float* A, float* B, float* C, int m, int k, int n);
float* allocateMatrix(int r, int c);
void printMatrix(float* ptr, int r, int c);
double parRunTime;
double seqRunTime;

/*-------------------------------------------------------------------*/
int main(int argc, char** argv) {
    int rank, size;
    
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float* A;
    float* B;
    float* C;

    B = allocateMatrix(k, n);
  
    if (rank == 0) {
        // Check if the number of processes divides m, k, and n
        if (m % size != 0) {
            printf("Error: the number of processes must divide m, k, and n\n");
            MPI_Finalize();
            exit(1);
        }

        // Allocate matrices A, B, and C
        A = allocateMatrix(m, k);
        C = allocateMatrix(m, n);

        // Populate matrices A and B with random numbers
        for (int i = 0; i < m * k; i++) {
            A[i] = (float)(rand()%1000);
        }
        for (int i = 0; i < k * n; i++) {
            B[i] = (float)(rand() % 1000);
        }

        printf("A Matrix:  \n");
        printMatrix(A, m, k);
        printf("\n");
        printf("B Matrix: \n");
        printMatrix(B, k, n);
        printf("\n");
    }

    // Perform parallel matrix multiplication
    parMatMul(A, B, C, m, k, n, size, MPI_COMM_WORLD);

    // Print matrix C
    if (rank == 0) {
        printf("Matrix C calculated in parallel:\n");
        printMatrix(C, m, n);
        printf("Process took %lf seconds\n", parRunTime);
        printf("\n");
        free(C);
    }

    if (rank == 0) {
        printf("Matrix C calculated squentially:\n");

        //start timer on the sequential execution
        double seqStartTime = MPI_Wtime();

        //Perform squential matrix multiplication
        seqMatMul(A, B, C, m, k, n);
        
        //stop timer on the sequential execution
        double seqStopTime = MPI_Wtime();

        //calculate sequential run time
        seqRunTime = seqStopTime - seqStartTime;

        printf("Process took %lf seconds\n", seqRunTime);

        // Free dynamically allocated memory
        free(A);
        free(B);
        free(C);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}/* main */


/*-------------------------------------------------------------------
 * Function:  seqMatMul
 * Purpose:   multiplies A and B sequentially. The function should 
 *            take A and B as inputs and return C as output.
 * In args:   A: A Matrix       m: number of rows in Matrix A, number of rows in Matrix C 
 *            B: B Matrix       k: number of collumns in Matrix A, number of rows in Matrix B
 *            C: C Matrix       n: number of collumns in Matrix B, number of collumns in Matrix C
 */
void seqMatMul(float* A, float* B, float* C, int m, int k,int n) {
    C = allocateMatrix(m, n);
    //Matrix calculation
    for (int i = 0; i < m * n; i++) {
        for (int j = 0; j < k; j++) {
            C[i] += A[((i / n) * k) + j] * B[(j * n) + i % n];
        }
    }

    printMatrix(C, m, n);
}/*seqMatMul*/


/*-------------------------------------------------------------------
 * Function:  seqMatMul
 * Purpose:   Matrix-Matrix Multiplication using MPI
 * In args:   A: A Matrix       m: number of rows in Matrix A, number of rows in Matrix C
 *            B: B Matrix       k: number of collumns in Matrix A, number of rows in Matrix B
 *            C: C Matrix       n: number of collumns in Matrix B, number of collumns in Matrix C
 *            size:   number of processes in comm
 *            comm:      communicator containing all processes calling
 *                       parMatMul
 */
void parMatMul(float* A, float* B, float* C, int m, int k, int n, int size, MPI_Comm comm) {
    int m_local = m / size;

    // Allocate memory for local matrices
    float* local_A = allocateMatrix(m_local, k);
    float* local_C = allocateMatrix(m_local, n);

    // Initialize matrix C to zero
    for (int i = 0; i < m_local; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
        }
    }

    // Broadcast B matrices to all processes
    MPI_Bcast(B, k * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter A matrices to all processes
    MPI_Scatter(A, m_local * k, MPI_FLOAT, local_A, m_local * k, MPI_FLOAT, 0, comm);
    
    //start timer on the parrallel execution
    double parStartTime = MPI_Wtime();

    //Matrix calculation
    for (int i = 0; i < m_local; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                local_C[i * n + j] += local_A[i * k + p] * B[p * n + j];
            }
        }
    }

    //stop timer on the parrallel execution
    double parStopTime = MPI_Wtime();

    //calculate parallel run time
    parRunTime = parStopTime - parStartTime;

    // Gather local C matrices into global C matrix
    MPI_Gather(local_C, m_local * n, MPI_FLOAT, C, m_local * n, MPI_FLOAT, 0, comm);

    // Free memory for local matrices
    free(local_A);
    free(local_C);
}/*parMatMu*/

/*-------------------------------------------------------------------
 * Function:  allocateMatrix
 * Purpose:   allocate space for a matrix in memory dynamically
 * In args:   r: number of rows in the matrix
 *            c: number of collumns in the matrix
 */
float* allocateMatrix(int r, int c) {
    float* ptr = malloc((r * c) * sizeof(float));
    return ptr;
}/*allocateMatrix*/

/*-------------------------------------------------------------------
 * Function:  printMatrix
 * Purpose:   print the matrix rows by rows
 * In args:   r: number of rows in the matrix
 *            c: number of collumns in the matrix
 *            ptr: the matrix to print
 */
void printMatrix(float* ptr, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.1f ", ptr[i * c + j]);
        }
        printf("\n");
    }
}/*printMatrix*/



#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define DECIMAL_NS 10

static int rank = 0;
static int countOfLines = 0;
static int countOfColumns = 0;
static MPI_Comm COMM_2D, COMM_COLUMNS, COMM_LINES;

void matrixMultiplication(const double *subA, const double *subB, double *subC, const int *N, const int *P){
    for (int i = 0; i < countOfLines; ++i){
        for (int j = 0; j < countOfColumns; ++j){
            subC[i * countOfColumns + j] = 0;
            for (int k = 0; k < N[1]; ++k){
                subC[i * countOfColumns + j] += subA[i * N[1] + k] * subB[k * countOfColumns + j];
            }
        }
    }
}

void getFullMatrix(double *C, double *subC, const int *N, const int *P){
    MPI_Datatype subC_t, subMat_t;
    MPI_Type_vector(countOfLines, countOfColumns, N[2], MPI_DOUBLE, &subMat_t);

    int blockLengths[2] = {1, 1};
    MPI_Datatype types[2] = {subMat_t, MPI_UB};
    long int displaceMents[2] = {0, (int)sizeof(double) * countOfColumns};

    MPI_Type_struct(2, blockLengths, displaceMents, types, &subC_t);
    MPI_Type_commit(&subC_t);

    int recvCounts[P[0] * P[1]];
    int displs[P[0] * P[1]];

    for (int i = 0; i < P[0]; ++i){
        for (int j = 0; j < P[1]; ++j) {
            recvCounts[i * P[1] + j] = 1;
            displs[i * P[1] + j] = i * P[1] * countOfLines + j;
        }
    }

    MPI_Gatherv(subC, countOfLines * countOfColumns, MPI_DOUBLE, C, recvCounts, displs, subC_t, 0, COMM_2D);

    MPI_Type_free(&subC_t);
}

double* initMatA(double *mat, int N, int M){
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < M; ++j){
            mat[i * M + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    return mat;
}


void initMatB(double *mat, int N, int M){
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < M; ++j){
            mat[i * M + j] = i * M + j;
        }
    }
}

void createLattice(const int *N, const int *P){
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, P, periods, 0, &COMM_2D);

    int remainDims[2] = {1, 0};
    MPI_Cart_sub(COMM_2D, remainDims, &COMM_COLUMNS);
    remainDims[0] = 0;
    remainDims[1] = 1;
    MPI_Cart_sub(COMM_2D, remainDims, &COMM_LINES);
    MPI_Comm_rank(COMM_2D, &rank);
}

bool preparationLattice(double *subA, double *subB, const int *N, const int *P){
    int sizeBlockMatA = countOfLines * N[1];
    int sizeBlockMatB = countOfColumns * N[1];

    double *A = NULL, *B = NULL;
    if (rank == 0){
        A = malloc(N[0] * N[1] * sizeof(double));
        if (!A){
            fprintf(stderr, "Bad allocation\n");
            return EXIT_FAILURE;
        }
        initMatA(A, N[0], N[1]);
        B = malloc(N[1] * N[2] * sizeof(double));
        if (!B){
            free(A);
            fprintf(stderr, "Bad allocation\n");
            return EXIT_FAILURE;
        }
        initMatB(B, N[1], N[2]);
    }

    int coords[2] = {0, 0};
    MPI_Cart_coords(COMM_2D, rank, 2, coords);

    if (coords[1] == 0){
        MPI_Scatter(A, sizeBlockMatA, MPI_DOUBLE, subA, sizeBlockMatA, MPI_DOUBLE, 0, COMM_COLUMNS);
    }

    MPI_Datatype subB_t, subMat_t;
    MPI_Type_vector(N[1], countOfColumns, N[2], MPI_DOUBLE, &subMat_t);
    int blockLengths[2] = {1, 1};
    long int displaceMents[2] = {0, (int)sizeof(double) * countOfColumns};
    MPI_Datatype types[2] = {subMat_t, MPI_UB};
    MPI_Type_struct(2, blockLengths, displaceMents, types, &subB_t);
    MPI_Type_commit(&subB_t);

    if (coords[0] == 0){
        int displs[P[1]];
        int counts[P[1]];
        for (int i = 0; i < P[1]; ++i){
            counts[i] = 1;
            displs[i] = i;
        }
        MPI_Scatterv(B, counts, displs, subB_t, subB, sizeBlockMatB, MPI_DOUBLE, 0, COMM_LINES);
    }

    if (rank == 0){
        free(A);
        free(B);
    }

    MPI_Bcast(subA, sizeBlockMatA, MPI_DOUBLE, 0, COMM_LINES);
    MPI_Bcast(subB, sizeBlockMatB, MPI_DOUBLE, 0, COMM_COLUMNS);

    MPI_Type_free(&subB_t);
    MPI_Type_free(&subMat_t);

    return EXIT_SUCCESS;
}

bool inputAndValidateData(char **argv, int *N, int *P){
    for (int i = 0; i < 3; ++i){
        long int tmp = strtol(argv[i + 1], NULL, DECIMAL_NS);
        if (tmp == 0 || tmp == LONG_MAX || tmp == LONG_MIN){
            fprintf(stderr, "Wrong arguments\n");
            return EXIT_FAILURE;
        } else {
            N[i] = (int)tmp;
        }
    }

    for (int i = 0; i < 2; ++i){
        long int tmp = strtol(argv[i + 4], NULL, DECIMAL_NS);
        if (tmp == 0 || tmp == LONG_MAX || tmp == LONG_MIN){
            fprintf(stderr, "Wrong arguments\n");
            return EXIT_FAILURE;
        } else if ((N[0] % tmp != 0 && i == 0) || (N[2] % tmp != 0 && i == 1)){
            fprintf(stderr, "Wrong arguments\n");
            return EXIT_FAILURE;
        } else {
            P[i] = (int)tmp;
        }
    }

    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (P[0] * P[1] != size) {
        fprintf(stderr, "Wrong arguments\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char **argv){
    if (argc < 6){
      fprintf(stderr, "Not enough arguments.\n");
      return EXIT_FAILURE;
    }

    int N[3] = {0, 0, 0};
    int P[2] = {0, 0};

    MPI_Init(&argc, &argv);

    if (inputAndValidateData(argv, N, P) == EXIT_FAILURE){
       return EXIT_FAILURE;
    }

    countOfLines = N[0] / P[0];
    int sizeBlockMatA = countOfLines * N[1];
    countOfColumns = N[2] / P[1];
    int sizeBlockMatB = countOfColumns * N[1];

    double *subA = malloc(sizeBlockMatA * sizeof(double));
    if (!subA){
        fprintf(stderr, "Bad allocation\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    bool error = 0;
    double *subB = malloc(sizeBlockMatB * sizeof(double));
    if (!subB){
        fprintf(stderr, "Bad allocation\n");
        error = EXIT_FAILURE;
    }

    double *subC = NULL;
    if (error != EXIT_FAILURE) {
        subC = malloc( countOfColumns * countOfLines * sizeof(double));
        if (!subC){
            fprintf(stderr, "Bad allocation\n");
            error = EXIT_FAILURE;
        }
    }

    double start = MPI_Wtime();

    createLattice(N, P);

    if (error != EXIT_FAILURE) {
        if (preparationLattice(subA, subB, N, P) == EXIT_FAILURE) {
            fprintf(stderr, "Bad allocation\n");
            error = EXIT_FAILURE;
        }
    }

    if (error != EXIT_FAILURE){
        matrixMultiplication(subA, subB, subC, N, P);
    }

    double *fullC = NULL;
    if (rank == 0 && error != EXIT_FAILURE){
        fullC = malloc(N[0] * N[2] * sizeof(double));
        if (!fullC){
            fprintf(stderr, "Bad allocation\n");
            return EXIT_FAILURE;
        }
    }

    if (error != EXIT_FAILURE){
        getFullMatrix(fullC, subC, N, P);
    }

    double end = MPI_Wtime();

    if (error != EXIT_FAILURE && rank == 0){
        printf("Time taken: %.0lf sec.\n", end - start);
    }

    free(subA);
    free(subB);
    free(subC);
    if (rank == 0){
        free(fullC);
    }

    MPI_Comm_free(&COMM_2D);
    MPI_Comm_free(&COMM_COLUMNS);
    MPI_Comm_free(&COMM_LINES);

    MPI_Finalize();

    return EXIT_SUCCESS;
}


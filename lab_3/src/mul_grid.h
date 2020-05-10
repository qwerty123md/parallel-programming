#ifndef LAB_1_SRC_MUL2D_H_
#define LAB_1_SRC_MUL2D_H_

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>

extern int rank;
extern int size;
extern MPI_Comm COMM_2D, COMM_COLUMNS, COMM_LINES;

typedef struct data {
	double *A;
	double *B;
	double *C;
	double *sub_A;
	double *sub_B;
	double *sub_C;
	int N[3];
	int P[2];
	int lines_in_block;
	int columns_in_block;
} data_t;

void mul_matrix(data_t *data);

void gather_matrix(data_t *data);

void create_grid(data_t *data);

#endif //LAB_1_SRC_MUL2D_H_

#include "mul_grid.h"

void mul_matrix(data_t *data) {
	for (int i = 0; i < data->lines_in_block; ++i) {
		for (int j = 0; j < data->columns_in_block; ++j) {
			data->sub_C[i * data->columns_in_block + j] = 0;
			for (int k = 0; k < data->N[1]; ++k) {
				data->sub_C[i * data->columns_in_block + j] += data->sub_A[i * data->N[1] + k] * data->sub_B[k * data->columns_in_block + j];
			}
		}
	}
}

void gather_matrix(data_t *data) {
	MPI_Datatype sub_C_t, tmp_t;
	MPI_Type_vector(data->lines_in_block, data->columns_in_block, data->N[2], MPI_DOUBLE, &tmp_t);

	int block_len[2] = {1, 1};
	MPI_Datatype types[2] = {tmp_t, MPI_UB};
	MPI_Aint displacements[2] = {0, sizeof(double) * data->columns_in_block};
	MPI_Type_struct(2, block_len, displacements, types, &sub_C_t);
	MPI_Type_commit(&sub_C_t);

	int counts[data->P[0] * data->P[1]];
	int displs[data->P[0] * data->P[1]];

	for (int i = 0; i < data->P[0]; ++i) {
		for (int j = 0; j < data->P[1]; ++j) {
			counts[i * data->P[1] + j] = 1;
			displs[i * data->P[1] + j] = i * data->lines_in_block * data->P[1] + j;
		}
	}

	MPI_Gatherv(data->sub_C, data->lines_in_block * data->columns_in_block, MPI_DOUBLE, data->C, counts, displs, sub_C_t, 0, COMM_2D);
	MPI_Type_free(&sub_C_t);
}

bool create_grid(data_t *data) {
	int periods[2] = {0, 0};
	int sub_comm1[2] = {1, 0};
	int sub_comm2[2] = {0, 1};

	MPI_Cart_create(MPI_COMM_WORLD, 2, data->P, periods, 0, &COMM_2D);
	MPI_Cart_sub(COMM_2D, sub_comm1, &COMM_COLUMNS);
	MPI_Cart_sub(COMM_2D, sub_comm2, &COMM_LINES);
	MPI_Comm_rank(COMM_2D, &rank);

	int size_block_A = data->lines_in_block * data->N[1];
	int size_block_B = data->columns_in_block * data->N[1];

	MPI_Datatype sub_B_t, tmp_t;
	MPI_Type_vector(data->N[1], data->columns_in_block, data->N[2], MPI_DOUBLE, &tmp_t);
	int block_len[2] = {1, 1};
	MPI_Aint displacements[2] = {0, sizeof(double) * data->columns_in_block};
	MPI_Datatype types[2] = {tmp_t, MPI_UB};
	MPI_Type_struct(2, block_len, displacements, types, &sub_B_t);
	MPI_Type_commit(&sub_B_t);

	int coords[2] = {0, 0};
	MPI_Cart_coords(COMM_2D, rank, 2, coords);

	if (coords[1] == 0) {
		MPI_Scatter(data->A, size_block_A, MPI_DOUBLE, data->sub_A, size_block_A, MPI_DOUBLE, 0, COMM_COLUMNS);
	}

	if (coords[0] == 0) {
		int displs[data->P[1]];
		int counts[data->P[1]];
		for (int i = 0; i < data->P[1]; ++i) {
			counts[i] = 1;
			displs[i] = i;
		}
		MPI_Scatterv(data->B, counts, displs, sub_B_t, data->sub_B, size_block_B, MPI_DOUBLE, 0, COMM_LINES);
	}

	MPI_Bcast(data->sub_A, size_block_A, MPI_DOUBLE, 0, COMM_LINES);
	MPI_Bcast(data->sub_B, size_block_B, MPI_DOUBLE, 0, COMM_COLUMNS);

	MPI_Type_free(&sub_B_t);
	MPI_Type_free(&tmp_t);

	return EXIT_SUCCESS;
}

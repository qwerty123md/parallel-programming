#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "cgm.h"

static int size, rank;
timers_t timers;
static double start, end;

static void add_vectors(const double *a, const double *b, size_t size, double *res) {
	start = MPI_Wtime();

	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] + b[i];
	}

	end = MPI_Wtime();
	timers.add_vect += end - start;
}

static void sub_vectors(const double *a, const double *b, size_t size, double *res) {

	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] - b[i];
	}
}

static void mul_matrix_vector(const double *mat, const double *vec, double *res, size_t size, size_t block_size) {
	start = MPI_Wtime();

	for (size_t i = 0; i < block_size; ++i) {
		double row_sum = 0.0;
		for (size_t j = 0; j < size; ++j) {
			row_sum += mat[i * size + j] * vec[j];
		}
		res[i] = row_sum;
	}

	end = MPI_Wtime();
	timers.mul_mat_vec += end - start;
}
static void mul_num_vector(double a, double *vec, size_t size, double *res) {
	start = MPI_Wtime();

	for (size_t i = 0; i < size; ++i) {
		res[i] = a * vec[i];
	}

	end = MPI_Wtime();
	timers.mul_num_vec += end - start;
}

static double scalar_mul(const double *a, const double *b, size_t size) {
	start = MPI_Wtime();

	double result = 0.0;
	for (size_t i = 0; i < size; ++i) {
		result += a[i] * b[i];
	}

	end = MPI_Wtime();
	timers.scalar_mul += end - start;
	return result;
}

static bool check_end_alg(const double *r, const data_t *data) {
	start = MPI_Wtime();

	double norm1 = 0.0;
	double norm2 = 0.0;

	for (size_t i = 0; i < data->size; ++i) {
		norm1 += r[i] * r[i];
		norm2 += data->result_vector[i] * data->result_vector[i];
	}
	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);

	if (norm2 == 0.0) {
		perror("Division by zero in check_end_alg");
		exit(EXIT_FAILURE);
	}

	end = MPI_Wtime();
	timers.check += end - start;
	return (norm1 / norm2 < EPSILON) ? true : false;
}

int do_magic(data_t *data, int argc, char **argv) {                                     //Conjugate Gradient Method

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int rank_source = (rank + size - 1) % size;
	size_t mat_size = data->size;
	size_t block_size = mat_size / size;

	double a = 0.0;
	double b = 0.0;
	double scalar_prev_r = 0.0;

	double *tmp = calloc(mat_size, sizeof(double));
	double *r = calloc(mat_size, sizeof(double));
	double *z = calloc(mat_size, sizeof(double));
	double *local_mat = malloc(mat_size * block_size * sizeof(double));
	double *local_x = malloc(block_size * sizeof(double));
	double *local_z = malloc(block_size * sizeof(double));
	double *local_tmp = calloc(block_size, sizeof(double));
	double *local_res = calloc(block_size, sizeof(double));

	if (!(tmp && r && z && local_mat && local_x && local_z && local_tmp && local_res)) {
		perror("calloc err");
		return EXIT_FAILURE;
	}

	MPI_Scatter(data->coef_matrix,mat_size * block_size, MPI_DOUBLE, local_mat,mat_size * block_size, MPI_DOUBLE,0, MPI_COMM_WORLD);
	MPI_Scatter(data->x_vector, block_size, MPI_DOUBLE, local_x, block_size, MPI_DOUBLE,0, MPI_COMM_WORLD);

	mul_matrix_vector(data->coef_matrix, data->x_vector, tmp, mat_size, mat_size);
	sub_vectors(data->result_vector, tmp, mat_size, r);
	memcpy(z, r, data->size * sizeof(double));

	MPI_Scatter(z, block_size, MPI_DOUBLE, local_z, block_size, MPI_DOUBLE,0, MPI_COMM_WORLD);

	while (!check_end_alg(r, data)) {
		memset(local_res,0,block_size * sizeof(double));

		for (int i = 0; i < size; ++i) {
			int offset = (rank + size - i) % size * block_size;
			for (size_t j = 0; j < block_size; ++j) {
				mul_matrix_vector(local_mat + j * mat_size + offset, local_z, local_tmp + j, block_size, 1);
			}
			add_vectors(local_res, local_tmp, block_size, local_res);
			MPI_Sendrecv_replace(local_z, block_size, MPI_DOUBLE,(rank + 1) % size,1, rank_source,1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		MPI_Allgather(local_res, block_size, MPI_DOUBLE, tmp, block_size, MPI_DOUBLE, MPI_COMM_WORLD);
		a = scalar_mul(r, r, mat_size) / scalar_mul(tmp, z, mat_size); //a

		scalar_prev_r = scalar_mul(r, r, mat_size);

		mul_num_vector(a, tmp, mat_size, tmp);                        //r
		sub_vectors(r, tmp, mat_size, r);

		mul_num_vector(a, local_z, block_size, local_tmp);            //x
		add_vectors(local_x, local_tmp, block_size, local_x);

		b = scalar_mul(r, r, mat_size) / scalar_prev_r;               //b

		mul_num_vector(b, local_z, block_size, local_z);              //z
		add_vectors(r + rank * block_size, local_z, block_size, local_z);

		MPI_Allgather(local_z, block_size, MPI_DOUBLE, z, block_size, MPI_DOUBLE, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allgather(local_x, block_size, MPI_DOUBLE, data->x_vector, block_size, MPI_DOUBLE, MPI_COMM_WORLD);

	free(tmp);
	free(r);
	free(z);
	free(local_mat);
	free(local_tmp);
	free(local_x);
	free(local_z);
	free(local_res);

	MPI_Finalize();

	return 0;
}

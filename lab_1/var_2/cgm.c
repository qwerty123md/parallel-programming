#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "cgm.h"

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

static void mul_matrix_vector(const double *mat, const double *vec, size_t size, double *res) {
	start = MPI_Wtime();

	for (size_t i = 0; i < size; ++i) {
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

int do_magic(data_t *data) {                                     //Conjugate Gradient Method
	double a = 0.0;
	double b = 0.0;
	double scalar_prev_r = 0;

	size_t size = data->size;

	double *tmp = calloc(size, sizeof(double));
	double *r = calloc(size, sizeof(double));
	double *z = calloc(size, sizeof(double));

	if (!(tmp && r && z)) {
		perror("calloc err");
		return EXIT_FAILURE;
	}

	mul_matrix_vector(data->coef_matrix, data->x_vector, size, tmp);
	sub_vectors(data->result_vector, tmp, size, r);
	memcpy(z, r, data->size * sizeof(double));

	while (!check_end_alg(r, data)) {
		mul_matrix_vector(data->coef_matrix, z, size, tmp);       //a
		a = scalar_mul(r, r, size) / scalar_mul(tmp, z, size);

		scalar_prev_r = scalar_mul(r, r, size);

		mul_num_vector(a, tmp, size, tmp);                        //r
		sub_vectors(r, tmp, size, r);

		mul_num_vector(a, z, size, tmp);                          //x
		add_vectors(data->x_vector, tmp, size, data->x_vector);

		b = scalar_mul(r, r, size) / scalar_prev_r;               //b

		mul_num_vector(b, z, size, z);                            //z
		add_vectors(r, z, size, z);
	}

	free(tmp);
	free(r);
	free(z);
	return 0;
}

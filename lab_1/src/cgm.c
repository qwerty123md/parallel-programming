#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "cgm.h"

static void add_vectors(const double *a, const double *b, size_t size, double *res) {
	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] + b[i];
	}
}

static void sub_vectors(const double *a, const double *b, size_t size, double *res) {
	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] - b[i];
	}
}

static void mul_matrix_vector(const double *mat, const double *vec, size_t size, double *res) {
	for (size_t i = 0; i < size; ++i) {
		double row_sum = 0;
		for (size_t j = 0; j < size; ++j) {
			row_sum += mat[i * size + j] * vec[j];
		}
		res[i] = row_sum;
	}
}
static void mul_num_vector(double a, double *vect, size_t size, double *res) {
	for (size_t i = 0; i < size; ++i) {
		res[i] = a * vect[i];
	}
}

static double scalar_mul(const double *a, const double *b, size_t size) {
	double result = 0;
	for (size_t i = 0; i < size; ++i) {
		result += a[i] * b[i];
	}
	return result;
}

static bool check_end_alg(const double *r, const data_t *data) {
	double norm1 = 0;
	double norm2 = 0;

	for (size_t i = 0; i < data->size; ++i) {
		norm1 += r[i] * r[i];
		norm2 += data->result_vector[i] * data->result_vector[i];
	}
	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);
	return (norm1 / norm2 < EPSILON) ? true : false;
}

int do_magic(data_t *data) {                                     //Conjugate Gradient Method
	double a = 0;
	double b = 0;
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

	while(!check_end_alg(r, data)) {
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

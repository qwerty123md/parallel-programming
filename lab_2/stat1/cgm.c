#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include "cgm.h"

timers_t timers;
static struct timespec start, end;

static void add_vectors(const double *a, const double *b, size_t size, double *res) {
	clock_gettime(CLOCK_REALTIME, &start);

	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] + b[i];
	}

	clock_gettime(CLOCK_REALTIME, &end);
	timers.add_vect += 1000000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
}

static void sub_vectors(const double *a, const double *b, size_t size, double *res) {
	for (size_t i = 0; i < size; ++i) {
		res[i] = a[i] - b[i];
	}
}

static void mul_matrix_vector(const double *mat, const double *vec, size_t size, double *res) {
	clock_gettime(CLOCK_REALTIME, &start);
	
	#pragma omp parallel for num_threads(4) schedule(static) //static dynamic guided
	for (size_t i = 0; i < size; ++i) {
		double row_sum = 0;
		for (size_t j = 0; j < size; ++j) {
			row_sum += mat[i * size + j] * vec[j];
		}
		res[i] = row_sum;
	}
	
	clock_gettime(CLOCK_REALTIME, &end);
	timers.mul_mat_vec += 1000000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
}
static void mul_num_vector(double a, double *vec, size_t size, double *res) {
	clock_gettime(CLOCK_REALTIME, &start);

	//#pragma omp parallel for
	for (size_t i = 0; i < size; ++i) {
		res[i] = a * vec[i];
	}

	clock_gettime(CLOCK_REALTIME, &end);
	timers.mul_num_vec += 1000000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
}

static double scalar_mul(const double *a, const double *b, size_t size) {
	clock_gettime(CLOCK_REALTIME, &start);

	double result = 0;
	//#pragma omp parallel for reduction(+: result)
	for (size_t i = 0; i < size; ++i) {
		result += a[i] * b[i];
	}
	
	clock_gettime(CLOCK_REALTIME, &end);
	timers.scalar_mul += 1000000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
	return result;
}

static bool check_end_alg(const double *r, const data_t *data) {
	clock_gettime(CLOCK_REALTIME, &start);

	double norm1 = 0;
	double norm2 = 0;

	for (size_t i = 0; i < data->size; ++i) {
		norm1 += r[i] * r[i];
		norm2 += data->result_vector[i] * data->result_vector[i];
	}
	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);

	clock_gettime(CLOCK_REALTIME, &end);
	timers.check += 1000000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
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

#ifndef LAB_1_SRC_CGM_H_
#define LAB_1_SRC_CGM_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define EPSILON (1E-25)

typedef struct timers {
	double mul_mat_vec;
	double scalar_mul;
	double mul_num_vec;
	double add_vect;
	double check;
	double all_magic;
} timers_t;

extern timers_t timers;

typedef struct data {
	double *coef_matrix;
	double *result_vector;
	double *x_vector;
	size_t size;
} data_t;

int do_magic(data_t *data);            //Conjugate Gradient Method

#endif //LAB_1_SRC_CGM_H_

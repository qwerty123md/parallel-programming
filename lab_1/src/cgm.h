#ifndef LAB_1_SRC_CGM_H_
#define LAB_1_SRC_CGM_H_

#include <stdlib.h>
#include <stdio.h>

#define EPSILON (1.0/100000)

typedef struct data {
	double *coef_matrix;
	double *result_vector;
	double *x_vector;
	size_t size;
} data_t;

int do_magic(data_t *data);            //Conjugate Gradient Method

#endif //LAB_1_SRC_CGM_H_

#include <time.h>
#include <stdint.h>
#include <string.h>

#include "cgm.h"

#define GET_RAND(min, max) (rand() % ((max) - (min) + 1) + (min))

static void print_solution(data_t *data) {
	for (size_t i = 0; i < data->size; i++) {
		printf("%lf | ", data->x_vector[i]);
	}
	putc('\n', stdout);
}

static int initial_alg(FILE *config, data_t *data) {
	srand(time(NULL));
	fscanf(config, "%zu", &data->size);

	data->result_vector = calloc(data->size, sizeof(double));
	data->x_vector = calloc(data->size, sizeof(double));
	data->coef_matrix = calloc(data->size * data->size, sizeof(double));

	if (!(data->result_vector && data->x_vector && data->coef_matrix)) {
		return EXIT_FAILURE;
	}
	for (size_t i = 0; i < data->size; ++i) {
		for (size_t j = 0; j < data->size; ++j) {
			if (fscanf(config, "%lf", &data->coef_matrix[i * data->size + j]) != 1) {
				return EXIT_FAILURE;
			}
		}
	}
	for (size_t i = 0; i < data->size; ++i) {
		if (fscanf(config, "%lf", &data->result_vector[i]) != 1) {
			return EXIT_FAILURE;
		}
	}
	for (size_t i = 0; i < data->size; ++i) {
		data->x_vector[i] = (double) GET_RAND(-10, 10);
		//memset(data->x_vector, 1, data->size);                            //don't touch. this for benchmark
	}
	return 0;
}

static void free_data(data_t *data, FILE *config) {
	free(data->x_vector);
	free(data->coef_matrix);
	free(data->result_vector);
	free(data);
	fclose(config);
}

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Bad arg. Please, use ./lab_1 config.txt\n"
		       "Where config.txt contain:\n"
		       "N\n"
		       "matrix N x N\n"
		       "vector N\n");
	}
	uint8_t err_code = 0;
	FILE *config = fopen(argv[1], "r");
	if (!config) {
		perror("Can't open config file");
		return EXIT_FAILURE;
	}

	data_t *data = calloc(1, sizeof(data_t));
	if (!data) {
		perror("Calloc err");
		fclose(config);
		return EXIT_FAILURE;
	}

	err_code = initial_alg(config, data);
	if (err_code) {
		perror("Problems with initialing alg \n{bad config or calloc err}");
		free_data(data, config);
		return err_code;
	}

	err_code = do_magic(data);
	if (err_code) {
		free_data(data, config);
		return err_code;
	}

	print_solution(data);

	free_data(data, config);
	return 0;
}

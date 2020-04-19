#include <stdint.h>
#include <string.h>

#include "cgm.h"

#define GET_RAND(min, max) (rand() % ((max) - (min) + 1) + (min))
static double start, end;

static void print_solution(data_t *data) {
	for (size_t i = 0; i < data->size; i++) {
		printf("%lf | ", data->x_vector[i]);
	}
	putc('\n', stdout);
}

static void dump_log() {
	FILE *out = fopen("log.txt", "a+");
	if (!out) perror("log open fall");
	fprintf(out,
	        "mul_matrix: %.3f\nscalar_mul: %.3f\nmul_num_vector: %.3f\nadd_vectors: %.3f\ncheck: %.3f\nall: %.3f\n\n",
	        timers.mul_mat_vec ,
	        timers.scalar_mul ,
	        timers.mul_num_vec ,
	        timers.add_vect ,
	        timers.check,
	        timers.all_magic );
	fclose(out);
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
		//data->x_vector[i] = (double) GET_RAND(-10, 10);
		memset(data->x_vector, 1, data->size);                            //don't touch. this for benchmark
	}
	return 0;
}

static void free_data(data_t *data, FILE *config) {
	free(data->x_vector);
	free(data->coef_matrix);
	free(data->result_vector);
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
	data_t data = {0};
	FILE *config = fopen(argv[1], "r");
	if (!config) {
		perror("Can't open config file");
		return EXIT_FAILURE;
	}

	err_code = initial_alg(config, &data);
	if (err_code) {
		perror("Problems with initialing alg \n{bad config or calloc err}");
		free_data(&data, config);
		return err_code;
	}

	start = MPI_Wtime();

	err_code = do_magic(&data, argc, argv);
	if (err_code) {
		free_data(&data, config);
		return err_code;
	}

	end = MPI_Wtime();
	timers.all_magic += end - start;

	print_solution(&data);
	dump_log();

	free_data(&data, config);
	return 0;
}

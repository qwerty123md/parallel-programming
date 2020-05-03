#include "mul_grid.h"
#include <errno.h>

#define GET_RAND(min, max) (rand() % ((max) - (min) + 1) + (min))
#define BASE 10

int rank = 0;
int size = 0;
MPI_Comm COMM_2D, COMM_COLUMNS, COMM_LINES;

static void init_matrix(double *mat, int N, int M) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			mat[i * M + j] = GET_RAND(-1000, 1000);
		}
	}
}

static bool check_args(char **argv, data_t *data) {
	for (int i = 0; i < 5; ++i) {
		int tmp = (int) strtol(argv[i + 1], NULL, BASE);
		if (tmp == 0 || errno == ERANGE) {
			perror("Bad argument\n");
			return EXIT_FAILURE;
		}
		if (i < 3) {
			data->N[i] = tmp;
		} else {
			data->P[i - 3] = tmp;
		}
	}

	if ((data->N[0] % data->P[0] != 0) || (data->N[2] % data->P[1]) != 0) {
		printf("Bad args. N must divide by P1, and K by P2\n");
		return EXIT_FAILURE;
	}

	if (data->P[0] * data->P[1] != size) {
		if (rank == 0) {
			printf("Cannot create a grid, wrong quantity threads\n");
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	return 0;
}

static void free_data(data_t *data) {
	if (rank == 0) {
		free(data->A);
		free(data->B);
		free(data->C);
	}

	free(data->sub_A);
	free(data->sub_B);
	free(data->sub_C);

	MPI_Comm_free(&COMM_2D);
	MPI_Comm_free(&COMM_COLUMNS);
	MPI_Comm_free(&COMM_LINES);
}

int main(int argc, char **argv) {

	if (argc != 6) {
		printf("Bad qty argumets. Expected 5 args, have: %d"
		       "Please, enter N,M,K and P1,P2\n", argc - 1);
		return EXIT_FAILURE;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	data_t data;

	if (check_args(argv, &data) == 1) {
		return EXIT_FAILURE;
	}

	if (rank == 0) {
		data.A = malloc(data.N[0] * data.N[1] * sizeof(double));
		data.B = malloc(data.N[1] * data.N[2] * sizeof(double));
		data.C = malloc(data.N[0] * data.N[2] * sizeof(double));
		if(!(data.A && data.B && data.C)){
			perror("Allocate error memory for matrix\n");
			free_data(&data);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		init_matrix(data.A, data.N[0], data.N[1]);
		init_matrix(data.B, data.N[1], data.N[2]);
	}

	data.lines_in_block = data.N[0] / data.P[0];
	data.columns_in_block = data.N[2] / data.P[1];

	int size_block_A = data.lines_in_block * data.N[1];
	int size_block_B = data.columns_in_block * data.N[1];

	data.sub_A = malloc(size_block_A * sizeof(double));
	data.sub_B = malloc(size_block_B * sizeof(double));
	data.sub_C = malloc(data.columns_in_block * data.lines_in_block * sizeof(double));

	if(!(data.sub_A && data.sub_B && data.sub_C)){
		perror("Allocate error memory for submatrix\n");
		free_data(&data);
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	double start = MPI_Wtime();

	create_grid(&data);
	mul_matrix(&data);
	gather_matrix(&data);

	double end = MPI_Wtime();

	if (rank == 0) {
		printf("\nTime taken: %.3f sec.\n", end - start);
	}

	free_data(&data);
	MPI_Finalize();

	return 0;
}

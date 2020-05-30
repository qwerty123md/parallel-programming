#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define E 1E-8
#define A 1E+5
#define N_X 300
#define N_Y 300
#define N_Z 300
#define DOWN 0
#define UP 1

static int size = 0;
static int rank = 0;

static const double hx = 2.0 / (N_X - 1);
static const double hy = 2.0 / (N_Y - 1);
static const double hz = 2.0 / (N_Z - 1);

static double f_real(double x, double y, double z) {
	return x * x + y * y + z * z;
}

static double p(double x, double y, double z) {
	return 6 - A * f_real(x, y, z);
}

static int check_end(double *f_prev, double *f_next, int lines) {
	for (int i = 0; i < lines; ++i) {
		for (int j = 0; j < N_Y; ++j) {
			for (int k = 1; k < N_Z; ++k) {
				if (fabs(f_next[i * N_Y * N_Z + j * N_Z + k] - f_prev[i * N_Y * N_Z + j * N_Z + k]) > E) {
					return 0;
				}
			}
		}
	}
	return 1;
}

static void accuracy_rating(const double *f, int lines, int offset) {
	double local_max = 0.0;
	for (int i = 1; i < lines; i++) {
		for (int j = 1; j < N_Y; j++) {
			for (int k = 1; k < N_Z; k++) {
				double tmp = fabs(f[i * N_Y * N_Z + j * N_Y + k] - f_real((i + offset) * hx, j * hy, k * hz));
				if (tmp > local_max) {
					local_max = tmp;
				}
			}
		}
	}
	double max = 0.0;
	MPI_Allreduce(&local_max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("Max delta: %lf\n", max);
	}
}

static void init(double *f, int lines, int offset) {
	for (int i = 0; i < lines; ++i) {
		for (int j = 0; j < N_Y; ++j) {
			for (int k = 0; k < N_Z; ++k) {
				if ((i == 0 && offset == 0) || (i == lines - 1 && offset == N_X - lines)
					|| (j == 0) || (j == N_Y - 1) || (k == 0) || (k == N_Z - 1)) {
					f[i * N_Y * N_Z + j * N_Z + k] = f_real((offset + i) * hx, j * hy, k * hz);
				}
			}
		}
	}
}

static void send_border(double *f_prev,
                        MPI_Request *send,
                        MPI_Request *recv,
                        double **border,
                        int lines) {
	if (rank != 0) {
		MPI_Isend(&(f_prev[0]), N_Y * N_Z, MPI_DOUBLE, rank - 1, DOWN, MPI_COMM_WORLD, &send[DOWN]);
		MPI_Irecv(border[DOWN], N_Y * N_Z, MPI_DOUBLE, rank - 1, UP, MPI_COMM_WORLD, &recv[UP]);
	}
	if (rank != size - 1) {
		MPI_Isend(&(f_prev[(lines - 1) * N_Y * N_Z]), N_Y * N_Z, MPI_DOUBLE, rank + 1, UP, MPI_COMM_WORLD,
		          &send[UP]);
		MPI_Irecv(border[UP], N_Y * N_Z, MPI_DOUBLE, rank + 1, DOWN, MPI_COMM_WORLD, &recv[DOWN]);
	}
}

static void calculate_other(const double *f_prev, double *f_next, int lines, int offset) {
	for (int i = 1; i < lines - 1; ++i) {
		for (int j = 1; j < N_Y - 1; ++j) {
			for (int k = 1; k < N_Z - 1; ++k) {
				double F_i =
					(f_prev[(i + 1) * N_Y * N_Z + j * N_Z + k] + f_prev[(i - 1) * N_Y * N_Z + j * N_Z + k]) / (hx * hx);
				double F_j =
					(f_prev[i * N_Y * N_Z + (j + 1) * N_Z + k] + f_prev[i * N_Y * N_Z + (j - 1) * N_Z + k]) / (hy * hy);
				double F_k =
					(f_prev[i * N_Y * N_Y + j * N_Z + (k + 1)] + f_prev[i * N_Y * N_Z + j * N_Z + (k - 1)]) / (hz * hz);
				f_next[i * N_Y * N_Z + j * N_Y + k] = (1 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + A)) * (F_i + F_j + F_k - p((i + offset) * hx, j * hy, k * hz));
			}
		}
	}
}

static void wait_border(MPI_Request *recv, MPI_Request *send) {
	if (rank != 0) {
		MPI_Wait(&send[DOWN], MPI_STATUS_IGNORE);
		MPI_Wait(&recv[UP], MPI_STATUS_IGNORE);
	}
	if (rank != size - 1) {
		MPI_Wait(&send[UP], MPI_STATUS_IGNORE);
		MPI_Wait(&recv[DOWN], MPI_STATUS_IGNORE);
	}
}

static void calculate_border(const double *f_prev,
                             double *f_next,
                             const double **border,
                             int offset,
                             int lines) {
	for (int j = 1; j < N_Y - 1; ++j) {
		for (int k = 1; k < N_Z - 1; ++k) {
			if (rank != 0) {
				int i = 0;
				double F_i = (f_prev[(i + 1) * N_Y * N_Z + j * N_Z + k] + border[DOWN][j * N_Z + k]) / (hx * hx);
				double F_j =
					(f_prev[i * N_Y * N_Z + (j + 1) * N_Z + k] + f_prev[i * N_Y * N_Z + (j - 1) * N_Z + k]) / (hy * hy);
				double F_k =
					(f_prev[i * N_Y * N_Z + j * N_Y + (k + 1)] + f_prev[i * N_Y * N_Z + j * N_Z + (k - 1)]) / (hz * hz);
				f_next[i * N_Y * N_Z + j * N_Z + k] = (1 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + A)) * (F_i + F_j + F_k - p((i + offset) * hx, j * hy, k * hz));
			}
			if (rank != size - 1) {
				int i = lines - 1;
				double F_i = (border[UP][j * N_Z + k] + f_prev[(i - 1) * N_Y * N_Z + j * N_Z + k]) / (hx * hx);
				double F_j =
					(f_prev[i * N_Y * N_Z + (j + 1) * N_Z + k] + f_prev[i * N_Y * N_Z + (j - 1) * N_Z + k]) / (hy * hy);
				double F_k =
					(f_prev[i * N_Y * N_Z + j * N_Z + (k + 1)] + f_prev[i * N_Y * N_Z + j * N_Z + (k - 1)]) / (hz * hz);
				f_next[i * N_Y * N_Z + j * N_Z + k] = (1 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + A)) * (F_i + F_j + F_k - p((i + offset) * hx, j * hy, k * hz));
			}
		}
	}
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	size_t lines = N_X / size;
	size_t offset = rank * lines;

	double *f_prev = calloc(lines * N_Y * N_Z, sizeof(double));
	double *f_next = calloc(lines * N_Y * N_Z, sizeof(double));
	double *border[2];
	border[0] = malloc(N_Y * N_Z * sizeof(double));
	border[1] = malloc(N_Y * N_Z * sizeof(double));
	if (!(f_prev && f_next && border[0] && border[1])) {
		perror("Alocation err");
		return EXIT_FAILURE;
	}
	int is_local_finished = 0;
	int is_finished = 0;
	MPI_Request send[2];
	MPI_Request recv[2];

	double start = MPI_Wtime();

	init(f_prev, lines, offset);
	init(f_next, lines, offset);

	while (!is_finished) {
		double *tmp = f_prev;
		f_prev = f_next;
		f_next = tmp;
		send_border(f_prev, send, recv, border, lines);
		calculate_other(f_prev, f_next, lines, offset);
		wait_border(recv, send);
		calculate_border(f_prev, f_next, border, offset, lines);
		is_local_finished = check_end(f_prev, f_next, lines);
		MPI_Allreduce(&is_local_finished, &is_finished, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	}
	accuracy_rating(f_next, lines, offset);
	double end = MPI_Wtime();

	if (rank == 0) {
		printf("Time : %.3lf\n", end - start);
	}
	free(border[0]);
	free(border[1]);
	free(f_prev);
	free(f_next);
	MPI_Finalize();
	return 0;
}

#include <stdlib.h>

#include "tasks_manager.h"

int rank;
int size;
pthread_mutex_t mutex;

int qty_tasks;
int *weight;

int main(int argc, char **argv) {
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_t request_thread;
	pthread_mutex_init(&mutex, NULL);

	if (!pthread_create(&request_thread, &attr, wait_request, NULL)) {
		perror("Could't create request_thread");
		return EXIT_FAILURE;
	}
	pthread_attr_destroy(&attr);

	set_qty_tasks();
	weight = calloc(qty_tasks, sizeof(int));
	if(!weight){
		perror("Allocation error");
		return EXIT_FAILURE;
	}

	double start = MPI_Wtime();

	tasks_manager();
	pthread_cancel(request_thread);

	double end = MPI_Wtime();
	if (rank == 0) {
		printf("Time: %lf\n", end - start);
	}

	pthread_mutex_destroy(&mutex);
	free(weight);
	MPI_Finalize();
	return 0;
}

#include "tasks_manager.h"
#include <math.h>

static int tasks_in_queue;
static int cur_task;

void set_qty_tasks() {
	qty_tasks = QTY_TASKS / size;
	if (rank == 0) {
		qty_tasks += QTY_TASKS % size;
	}
}

static void do_task() {
	pthread_mutex_lock(&mutex);
	tasks_in_queue = qty_tasks;
	cur_task = 0;
	while (tasks_in_queue != 0) {
		tasks_in_queue--;
		int tmp = cur_task;
		pthread_mutex_unlock(&mutex);

		double accum = 0;
		for (int i = 0; i < weight[tmp]; ++i) {
			accum += log2(pow(3, 13) - i * cos(2 * 3.14 * i));
		}
		pthread_mutex_lock(&mutex);
		cur_task++;
	}
	pthread_mutex_unlock(&mutex);
}

void *tasks_manager() {
	for (int i = 0; i < QTY_WAVES; ++i) {
		for (int j = 0; j < qty_tasks; ++j) {
			weight[j] = (rank + 1 - (i % size)) * 10000 * (int) (log(j) + 1);
		}
		double start = MPI_Wtime();
		int is_list_full;
		do_task();
		do {
			is_list_full = 0;
			for (int j = 0; j < size - 1; ++j) {
				int next_proc = (rank + 1 + j) % size;
				MPI_Send(&rank, 1, MPI_INT, next_proc, 0, MPI_COMM_WORLD);
				MPI_Recv(&qty_tasks, 1, MPI_INT, next_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (qty_tasks) {
					MPI_Recv(weight, qty_tasks, MPI_INT, next_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					do_task();
					is_list_full = 1;
				}
			}
		} while (is_list_full);
		double end = MPI_Wtime();
		printf("Time %d of process %d: %.2lf\n", i, rank, end - start);
		set_qty_tasks();
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

void *wait_request() {
	int stub;
	MPI_Status status;
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
	while (1) {
		MPI_Recv(&stub, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		pthread_mutex_lock(&mutex);
		int send_task = 0;
		if (tasks_in_queue > 2) {
			send_task = tasks_in_queue / 2;
			MPI_Send(&send_task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
			MPI_Send(&weight[cur_task + 1], send_task, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
			cur_task += send_task;
			tasks_in_queue -= send_task;
		} else {
			MPI_Send(&send_task, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
		}
		pthread_mutex_unlock(&mutex);
	}
}

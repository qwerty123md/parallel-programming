#ifndef LAB_5_SRC_TASKS_MANAGER_H_
#define LAB_5_SRC_TASKS_MANAGER_H_

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>

#define QTY_TASKS 100
#define QTY_WAVES 10

extern int rank;
extern int size;
extern pthread_mutex_t mutex;

extern int qty_tasks;
extern int *weight;

void *wait_request();

void *tasks_manager();

void set_qty_tasks();

#endif //LAB_5_SRC_TASKS_MANAGER_H_

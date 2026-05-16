#ifndef DATASET_H
#define DATASET_H

#include "tree.h"

#define MAX_LEN_LINE 4096 //макимальная длина строки

typedef struct {                        //структура датасета
    double **X; //матрица признаков

    int *y_class; //массив классов для классификации
    double *y_reg; //массив значений для регресии

    int n_objects; //количество объектов
    int n_features; //количество признаков
    int num_classes; //количество классов
} Dataset;

Dataset *load_dataset(const char *filename, Task_Type task_type); //загрузка датасета

void free_dataset(Dataset *ds); //освобождение памяти датасета

void split_dataset(const Dataset *src, double test_ratio,
                   Dataset **train_out, Dataset **test_out);   //разделение датасета на train (обучение) и test (проверка)

#endif

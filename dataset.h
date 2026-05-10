#ifndef DATASET_H
#define DATASET_H

#define MAX_LEN_LINE 4096

typedef struct {                        //структура датасета
    double **X; //матрица признаков
    int *y; //метки классов
    int n_objects; //количество объектов
    int n_features; //количество признаков 
    int num_classes; //количество классов
} Dataset;

Dataset *load_dataset(const char *filename); //загрузка датасета

void free_dataset(Dataset *ds); //освобождение памяти датасета

void split_dataset(   //разделение датасета на train (обучение) и test (проверка)
    const Dataset *src,
    double test_ratio,
    Dataset **train_out,
    Dataset **test_out
);

#endif
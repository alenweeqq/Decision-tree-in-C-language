#ifndef FOREST_H
#define FOREST_H

#include "tree.h"
#include "dataset.h"

typedef struct {                            //лес
    Node_Tree **trees; //массив указательней на деревья
    int n_trees; //количество деревьев
    Params_Tree params; //параметры деревьев
    int num_classes; //количество классов
} Forest;

Forest *train_forest(double **X, int *y, int n_objects, int n_features,                  //создание леса
                            int n_trees, Params_Tree params);

int predict_forest(Forest *forest, const double *x); // предсказание леса

double accuracy_forest(Forest *forest, Dataset *ds);  //точность модели/ леса

void free_forest(Forest *forest); //освобождение леса

void save_forest(Forest *forest, const char *filename); //сохранение леса

void print_model_info(Forest *forest); //отображение информации

#endif
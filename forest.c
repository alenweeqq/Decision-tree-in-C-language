#include "forest.h"
#include "tree.h"
#include "dataset.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

Forest *train_forest(double **X, int *y, int n_objects, int n_features,                  //создание леса
                            int n_trees, Params_Tree params) {
    Forest *forest = (Forest *)xcalloc(1, sizeof(Forest));
    forest->trees = (Node_Tree **)xcalloc(n_trees, sizeof(Node_Tree *)); //инициализация
    forest->n_trees = n_trees;
    forest->params = params;
    forest->num_classes = params.num_classes;

    for (int t = 0; t < n_trees; t++) {
        double **X_dt = (double **)xmalloc(n_objects * sizeof(double *));
        int *y_dt = (int *)xmalloc(n_objects * sizeof(int));

        for (int i = 0; i < n_objects; i++) {
            int idx = rand() % n_objects; // случайно выбираем объект
            X_dt[i] = X[idx]; // его признак 
            y_dt[i] = y[idx]; // и класс
        }

        forest->trees[t] = build_tree(X_dt, y_dt, n_objects, n_features, 0, params); //строим дерево

        free(X_dt);
        free(y_dt);
    }

    return forest;
}

int predict_forest(Forest *forest, const double *x) {                                 // предсказание леса
    int *votes = (int *)xcalloc(forest->num_classes, sizeof(int)); //массив для голосов

    for (int t = 0; t < forest->n_trees; t++) { // проход по деревьям
        int pred = predict_tree(forest->trees[t], x);
        if (pred >= 0 && pred < forest->num_classes) { //получаем голос от каждого дерева, смотрим по диапозону и записываем в массив
            votes[pred]++;
        }
    }

    int best_class = 0; // лучший класс
    for (int c = 1; c < forest->num_classes; c++) {
        if (votes[c] > votes[best_class]) { // рассматриваемкаждый класс и смотрим какого больше
            best_class = c;
        }
    }

    free(votes);
    return best_class;
}

double accuracy_forest(Forest *forest, Dataset *ds) {                //точность модели/ леса
    int correct = 0; //счетчик соответсвия пердволожившего класса и настоящего класса
    for (int i = 0; i < ds->n_objects; i++) {
        int pred = predict_forest(forest, ds->X[i]);
        if (pred == ds->y[i]) correct++;
    }
    return (double)correct / ds->n_objects; //количество соответсвия на общее число
}

void free_forest(Forest *forest) {               //освобождение леса
    if (!forest) return;
    for (int i = 0; i < forest->n_trees; i++) { //освобождаем каждое дерево
        free_tree(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}

void save_forest(Forest *forest, const char *filename) {                         //сохранение леса
    FILE *f = fopen(filename, "w");
    if (!f) emergency_exit("Couldn't open the file"); //обработка ошибки

    fprintf(f, "%d %d\n", //сохраняем
            forest->n_trees, //количество деревьев
            forest->num_classes); //количество классов

    fprintf(f, "%d %d %d\n", //параметры обучения 
            forest->params.max_depth, //максимальная глубина
            forest->params.min_objects_split, //минимальное количество объектов для разделения
            forest->params.num_features_to_try); //количество случайного выбора признаков 

    for (int i = 0; i < forest->n_trees; i++) {
        save_tree(f, forest->trees[i]);
        fprintf(f, "#\n"); // разеделение дереьвьев
    }

    fclose(f);
}

void print_model_info(Forest *forest) {
    printf("\n--- INFORMATION ABOUT THE MODEL ---\n");
    printf("Trees: %d\n", forest->n_trees);
    printf("Classes: %d\n", forest->num_classes);
    printf("Max depth: %d\n", forest->params.max_depth);
    printf("Min objects split: %d\n",
         forest->params.min_objects_split);
    printf("Features per split: %d\n", forest->params.num_features_to_try);
    printf("--- END ---\n\n");
}
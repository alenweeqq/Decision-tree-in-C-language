#include "forest.h"
#include "tree.h"
#include "dataset.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

Forest *train_forest(double **X, int *y, int n_objects, int n_features,                         //создание леса классификации
                     int n_trees, Params_Tree params){
    Forest *forest = (Forest *)xcalloc(1, sizeof(Forest));

    forest->trees = (Node_Tree **)xcalloc(n_trees, sizeof(Node_Tree *));  //выделение памяти под деревья 
    forest->n_trees = n_trees; //инициализация параметров
    forest->params = params;
    forest->num_classes = params.num_classes;

    for (int t = 0; t < n_trees; t++) {
        double **X_dt = (double **)xmalloc( n_objects * sizeof(double *)); //признаки
        int *y_dt = (int *)xmalloc( n_objects * sizeof(int)); //классы

        for (int i = 0; i < n_objects; i++) {
            int idx = rand() % n_objects; // случайно выбираем объект
            X_dt[i] = X[idx]; // его признаки
            y_dt[i] = y[idx]; // и класс
        }

        forest->trees[t] = build_tree( X_dt, y_dt, n_objects, n_features, 0, params); //строим дерево

        free(X_dt);
        free(y_dt);
    }

    return forest;
}

Forest *train_regression_forest(double **X, double *y, int n_objects,                           //создание леса регрессии
                                int n_features, int n_trees, Params_Tree params){
    Forest *forest = xcalloc(1, sizeof(Forest));
    forest->trees = xcalloc(n_trees, sizeof(Node_Tree *));
    forest->n_trees = n_trees; //инициализация
    forest->params = params;

    for (int t = 0; t < n_trees; t++) {
        double **X_dt = xmalloc( n_objects * sizeof(double *)); //признаки
        double *y_dt = xmalloc( n_objects * sizeof(double)); //значения

        for (int i = 0; i < n_objects; i++) {
            int idx = rand() % n_objects; // берем рандомный объект
            X_dt[i] = X[idx];
            y_dt[i] = y[idx];
        }

        forest->trees[t] = build_regression_tree( X_dt, y_dt, n_objects,  //построение дерева
                n_features, 0, params);

        free(X_dt);

        free(y_dt);
    }

    return forest;
}

int predict_forest(Forest *forest, const double *x){                                // предсказание леса классификации

    int *votes = (int *)xcalloc( forest->num_classes, sizeof(int)); //массив для голосов

    for (int t = 0; t < forest->n_trees; t++) {
        int pred = predict_tree( forest->trees[t], x);
        //получаем голос от каждого дерева
        if (pred >= 0 && pred < forest->num_classes)
        {
            votes[pred]++;
        }
    }

    int best_class = 0; // лучший класс

    for (int c = 1; c < forest->num_classes; c++){
        //ищем максимальное число голосов
        if (votes[c] > votes[best_class])
        {
            best_class = c;
        }
    }
    free(votes);
    return best_class;
}

double predict_regression_forest(Forest *forest, const double *x){                                 // предсказание леса регрессии
    double sum = 0.0; //накопитель
    for (int t = 0; t < forest->n_trees; t++) {
        sum += predict_regression_tree( forest->trees[t], x); //накопление пердсказания
    }

    return sum / forest->n_trees; //берем среднее значение
}

double accuracy_forest(Forest *forest, Dataset *ds){                                           //точность модели/ леса классификации
    int correct = 0; //счетчик соответсвия предсказанного класса и настоящего класса

    for (int i = 0; i < ds->n_objects; i++){
        int pred = predict_forest( forest, ds->X[i]);

        if (pred == ds->y_class[i])
        {
            correct++;
        }
    }

    return (double)correct / ds->n_objects; //доля правильных ответов
}

double mse_forest(Forest *forest, Dataset *ds){                                                    //точность леса регрессии
    double mse = 0.0;

    for (int i = 0; i < ds->n_objects; i++){
        double pred = predict_regression_forest( forest, ds->X[i]);
        double diff = pred - ds->y_reg[i]; //отклонение
        mse += diff * diff;
    }
    return mse / ds->n_objects;
}

void free_forest(Forest *forest){                                                              //освобождение леса
    if (!forest)
        return;
    for (int i = 0; i < forest->n_trees; i++)
    {
        free_tree( forest->trees[i]); //освобождаем каждое дерево
    }

    free(forest->trees);
    free(forest);
}

void save_forest(Forest *forest, const char *filename){                                          //сохранение леса
    FILE *f = fopen(filename, "w");
    if (!f) emergency_exit( "Couldn't open the file");

    fprintf(f, "%d %d\n",
            forest->n_trees, //количество деревьев
            forest->num_classes); //количество классов

    fprintf(f, "%d %d %d\n",
            forest->params.max_depth, //максимальная глубина
            forest->params.min_objects_split, //минимум объектов
            forest->params.num_features_to_try); //случайные признаки

    for (int i = 0; i < forest->n_trees; i++){
        save_tree(f, forest->trees[i], forest->params.task_type);

        fprintf(f, "#\n"); //разделение деревьев
    }

    fclose(f);
}

void print_model_info(Forest *forest){
    printf("\n--- INFORMATION ABOUT THE MODEL ---\n");
    printf("Trees: %d\n", forest->n_trees);
    printf("Classes: %d\n", forest->num_classes);
    printf("Max depth: %d\n", forest->params.max_depth);
    printf("Min objects split: %d\n", forest->params.min_objects_split);
    printf("Features per split: %d\n", forest->params.num_features_to_try);
    printf("Task type: ");

    if (forest->params.task_type == TASK_CLASSIFICATION)
    {
        printf("classification\n");
    } else {
        printf("regression\n");
    }

    printf("--- END ---\n\n");
}

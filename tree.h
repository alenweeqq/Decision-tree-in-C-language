#ifndef TREE_H
#define TREE_H

#include <stdio.h>

typedef enum {                          //перечисление критерия разделения
    CRITERION_GINI, // джини
    CRITERION_ENTROPY //энтропия
} Criterion;

typedef enum {                           //перечисление типов задачи
    TASK_CLASSIFICATION, //классификация
    TASK_REGRESSION //регрессия
} Task_Type;


typedef struct Node_Tree {                 //структура узла дерева
    int node_leaf; // узел = 0, лист = 1
    int predicted_class; //какой класс предсказывает
    double predicted_value; //предесказанное значание
    int feature_index; //признак, по которому делим
    double limit; //предел сравнения
    struct Node_Tree *left; //левый ребенок
    struct Node_Tree *right; //правый ребенок
} Node_Tree;

typedef struct {                            //парметры дерева
    int max_depth; //макисмальная глубина
    int min_objects_split; //минимум объектов для разделения
    int num_features_to_try; //количество случайно выбранных признаков
    int num_classes; // количество классов
    Criterion criterion; //критерий разделения
    Task_Type task_type; //тип задачи
} Params_Tree;

Node_Tree *build_tree(double **X, int *y, int n_objects, int n_features,
                            int depth, Params_Tree params);                    //рекурсивное построение дерева решений (классификация)

int predict_tree(Node_Tree *root, const double *x);   //предсказание класса                          //предсказание дерева

Node_Tree *build_regression_tree(double **X, double *y, int n_objects,          //построение дерева регресии
    int n_features, int depth, Params_Tree params);

double predict_regression_tree(Node_Tree *root, //предсказание значения регрессии
    const double *x);

void free_tree(Node_Tree *root); //освобождение дерева

void print_tree(Node_Tree *node, int depth, Task_Type task_type);       //ASCII вывод дерева

void save_tree(FILE *f, Node_Tree *node, Task_Type task_type);      //сохранение дерева

void export_tree_to_dot(Node_Tree *root, const char *filename,
                        Task_Type task_type);                       //сохранение дерева в .dot

void export_tree_to_png(Node_Tree *root, const char *dot_filename,
                        const char *png_filename, Task_Type task_type); //сохранение дерева в png

#endif

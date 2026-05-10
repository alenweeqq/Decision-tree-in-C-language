#ifndef TREE_H
#define TREE_H

#include <stdio.h>

typedef struct Node_Tree {                 //структура узла дерева
    int node_leaf; // узел = 0, лист = 1
    int predicted_class; //какой класс предсказывает
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
} Params_Tree;

Node_Tree *build_tree(double **X, int *y, int n_objects, int n_features,
                            int depth, Params_Tree params);                 //рекурсивное построение дерева решений

int predict_tree(Node_Tree *root, const double *x);  //предсказание дерева

void free_tree(Node_Tree *root); //освобождение дерева

void print_tree(Node_Tree *node, int depth); //визуализация дерева

void save_tree(FILE *f, Node_Tree *node);               //сохранение дерева

#endif
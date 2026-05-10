#include "tree.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>

static int majority_class(const int *y, int n_objects, int num_classes) {       //нахождение самого частого класса среди объектов
    int *counts = (int *)xcalloc(num_classes, sizeof(int)); //счетчики классов
    for (int i = 0; i < n_objects; i++) {
        if (y[i] >= 0 && y[i] < num_classes) { //проверка на соответсвие класса
            counts[y[i]]++; //счет
        }
    }

    int best_class = 0;
    for (int c = 1; c < num_classes; c++) {
        if (counts[c] > counts[best_class]) { //нахождение лучшего( который больше встретился)
            best_class = c;
        }
    }

    free(counts);
    return best_class;
}

static int all_same_class(const int *y, int n_objects) { //проверка, одно класса ли все объекты,
    for (int i = 1; i < n_objects; i++) {              //чтобы предотвратить рост дерева и дальнейшие ошибки
        if (y[i] != y[0]) return 0; 
    }
    return 1;
}

static double counter_gini(const int *y, int n_objects, int num_classes) {          //нахождение коэфицента  неравномерного распределения
    if (n_objects == 0) return 0.0; //случай отсутвия объектов

    int *counts = (int *)xcalloc(num_classes, sizeof(int)); 
    for (int i = 0; i < n_objects; i++) {
        counts[y[i]]++;   //считаем сколько раз встратились те или иные классы
    }

    double gini = 1.0;
    for (int c = 0; c < num_classes; c++) {
        double p = (double)counts[c] / (double)n_objects; 
        gini -= p * p; //gini = 1 - ∑_(c=1)^K p_c^2
    }

    free(counts);
    return gini;
}

static Node_Tree *create_leaf(const int *y, int n_objects, int num_classes) {       //создание листа
    Node_Tree *node = (Node_Tree *)xcalloc(1, sizeof(Node_Tree)); // создаем узел
    node->node_leaf = 1; //флаг листа
    node->predicted_class = majority_class(y, n_objects, num_classes); //записываем класс
    return node;
}

static void choose_random_features(int n_features, int m, int *out) {           //выбор первых случайных m признаков
    int *all = (int *)xmalloc(n_features * sizeof(int));
    for (int i = 0; i < n_features; i++) all[i] = i; //выписываем признаки
    uniform_mixing(all, n_features); //перемешиваем

    if (m <= 0 || m > n_features) m = n_features; //проверка диапозона m, в случае несоответетвия берем все признаки
    for (int i = 0; i < m; i++) out[i] = all[i];

    free(all);
}

Node_Tree *build_tree(double **X, int *y, int n_objects, int n_features,
                            int depth, Params_Tree params) {                             //рекурсивное построение дерева решений
    if (n_objects == 0) return NULL; //отсутствие объектов

    if (depth >= params.max_depth ||
        n_objects < params.min_objects_split ||
        all_same_class(y, n_objects)) {
        return create_leaf(y, n_objects, params.num_classes); //остановка дерева и создание листа
    }

    int m = params.num_features_to_try; //количесво признаков
    if (m <= 0 || m > n_features) m = n_features;

    int *features = (int *)xmalloc(m * sizeof(int));
    choose_random_features(n_features, m, features); //случайно выбираем признак

    double best_gini = 1e18; //берем неудачное значение для дальнеших сравнений(инициализация)
    int best_feature = -1; //лучший признак не найден
    double best_limit = 0.0; //лучший порог

    for (int fi = 0; fi < m; fi++) {
        int feature = features[fi]; //перебираем признаки

        for (int i = 0; i < n_objects; i++) {
            double limit = X[i][feature]; //берем значение как порог

            int left_count = 0, right_count = 0; //счетчики по разделению по порогу
            for (int j = 0; j < n_objects; j++) {
                if (X[j][feature] < limit) left_count++;
                else right_count++;
            }

            if (left_count == 0 || right_count == 0) continue; // защита от плохого разделения

            int *y_left = (int *)xmalloc(left_count * sizeof(int)); //классы левый части
            int *y_right = (int *)xmalloc(right_count * sizeof(int)); //классы правой части

            int li = 0, ri = 0; //индексы
            for (int j = 0; j < n_objects; j++) {
                if (X[j][feature] < limit) y_left[li++] = y[j]; //первый проход для нахождение разделения
                else y_right[ri++] = y[j]; //вписываем по частям классы
            }

            double g_left = counter_gini(y_left, left_count, params.num_classes); //считаем на каждую часть gini 
            double g_right = counter_gini(y_right, right_count, params.num_classes);

            double general_gini = ((double)left_count / n_objects) * g_left +  //общий gini узла посла рапределения
                ((double)right_count / n_objects) * g_right;

            if (general_gini < best_gini) { //рассматриваем лучшее разделение
                best_gini = general_gini;
                best_feature = feature;
                best_limit = limit;
            }

            free(y_left);
            free(y_right);
        }
    }

    free(features);

    if (best_feature == -1) { //разделениене не нашлось
        return create_leaf(y, n_objects, params.num_classes); //создаем лист
    }

    int left_count = 0, right_count = 0;
    for (int i = 0; i < n_objects; i++) {  
        if (X[i][best_feature] < best_limit) left_count++; //второй проход разделения для постанивки
        else right_count++;
    }

    if (left_count == 0 || right_count == 0) {
        return create_leaf(y, n_objects, params.num_classes); //остутвие разделения
    }

    double **X_left = (double **)xmalloc(left_count * sizeof(double *)); //признаки частей
    double **X_right = (double **)xmalloc(right_count * sizeof(double *));
    int *y_left = (int *)xmalloc(left_count * sizeof(int)); //классы частей
    int *y_right = (int *)xmalloc(right_count * sizeof(int));

    int li = 0, ri = 0;
    for (int i = 0; i < n_objects; i++) {
        if (X[i][best_feature] < best_limit) { //разделяем
            X_left[li] = X[i];
            y_left[li] = y[i];
            li++;
        } else {
            X_right[ri] = X[i];
            y_right[ri] = y[i];
            ri++;
        }
    }

    Node_Tree *node = (Node_Tree *)xcalloc(1, sizeof(Node_Tree));
    node->node_leaf = 0; //создаем узел
    node->predicted_class = majority_class(y, n_objects, params.num_classes);
    node->feature_index = best_feature;
    node->limit = best_limit;

    node->left = build_tree(X_left, y_left, left_count, n_features, depth + 1, params); //идем рекурсивно
    node->right = build_tree(X_right, y_right, right_count, n_features, depth + 1, params);

    free(X_left);
    free(X_right);
    free(y_left);
    free(y_right);

    return node;
}

int predict_tree(Node_Tree *root, const double *x) {                                 //предсказание дерева
    Node_Tree *cur = root; //узел прохода (текщий), начинаем с корня
    while (cur && !cur->node_leaf) { 
        if (x[cur->feature_index] < cur->limit) cur = cur->left; //проход
        else cur = cur->right;
    }
    return cur ? cur->predicted_class : -1;
}

void free_tree(Node_Tree *root) {        //освобождение дерева
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

void print_tree(Node_Tree *node, int depth) {                     //визуализация дерева
    if (!node) return;

    for (int i = 0; i < depth; i++) { //отсупы
        printf("    ");
    }

    if (node->node_leaf) { //лист
        printf("class = %d\n", node->predicted_class);
        return;
    }

    printf("[X[%d] < %.3f]\n", node->feature_index, node->limit); //узел: признак и порог

    for (int i = 0; i < depth; i++) printf("    "); //левая ветка
    printf("|--- yes:\n");
    print_tree(node->left, depth + 1);

    for (int i = 0; i < depth; i++) printf("    "); // правая ветка
    printf("|___ no:\n");
    print_tree(node->right, depth + 1);
}

void save_tree(FILE *f, Node_Tree *node) {               //сохранение дерева
    if (!node) return;

    if (node->node_leaf) {
        fprintf(f, "L %d\n", node->predicted_class); // лист и класс
        return;
    }

    fprintf(f, "N %d %lf\n", node->feature_index, node->limit); //узел с признаком и порогом

    save_tree(f, node->left); //DFS, рекусрсивно в глубину обходим левую 
    save_tree(f, node->right); //и правую часть

    fprintf(f, "E\n");
}

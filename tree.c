#include "tree.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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
        counts[y[i]]++;   //считаем сколько раз встратились различные классы
    }

    double gini = 1.0;
    for (int c = 0; c < num_classes; c++) {
        double p = (double)counts[c] / (double)n_objects; 
        gini -= p * p; //gini = 1 - ∑_(c=1)^K p_c^2
    }

    free(counts);
    return gini;
}

static double counter_entropy(const int *y, int n_objects, int num_classes) {              //энтропия 
    if (n_objects == 0)
        return 0.0;
    int *counts = xcalloc(num_classes, sizeof(int)); //счетчик классов

    for (int i = 0; i < n_objects; i++) {
        counts[y[i]]++; 
    }
    double entropy = 0.0;
    for (int c = 0; c < num_classes; c++) {
        if (counts[c] == 0) //проверка на исключения
            continue;
        double p = (double)counts[c] / n_objects; //вероятность класса
        entropy -= p * log2(p); //энтропия = -∑ pi log 2 pi
    }
    free(counts);
    return entropy;
}

static double mean_value(const double *y, int n_objects) {                     //вычисление среднего значения для регрессии
    double sum = 0.0;
    for (int i = 0; i < n_objects; i++) {
        sum += y[i]; //накопительная сумма 
    }
    return sum / n_objects;
}

static double counter_mse(const double *y, int n_objects) {                 // критерий качества разделения (среднеквадратичная ошибка)
    if (n_objects == 0) //случай отсутсвия объектов
        return 0.0;
    double mean = mean_value(y, n_objects); //среднее значение
    double mse = 0.0;
    for (int i = 0; i < n_objects; i++) {
        double diff = y[i] - mean; //отклоление от среднего
        mse += diff * diff; //накопление
    }
    return mse / n_objects;
}

static Node_Tree *create_leaf(const int *y, int n_objects, int num_classes) {       //создание листа для классификации
    Node_Tree *node = (Node_Tree *)xcalloc(1, sizeof(Node_Tree)); // создаем узел
    node->node_leaf = 1; //флаг листа
    node->predicted_class = majority_class(y, n_objects, num_classes); //записываем класс
    return node;
}

static void choose_random_features(int n_features, int m, int *out) {           //выбор случайных m признаков
    int *all = (int *)xmalloc(n_features * sizeof(int));
    for (int i = 0; i < n_features; i++) all[i] = i; //выписываем признаки
    uniform_mixing(all, n_features); //перемешиваем

    if (m <= 0 || m > n_features) m = n_features; //проверка диапозона m, в случае несоответетвия берем все признаки
    for (int i = 0; i < m; i++) out[i] = all[i];

    free(all);
}

static Node_Tree *create_regression_leaf(const double *y, int n_objects) {                      //создание листа для регрессии
    Node_Tree *node = xcalloc(1, sizeof(Node_Tree));
    node->node_leaf = 1;
    node->predicted_value = mean_value(y, n_objects); //среднее значение
    return node;
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

    double best_quality;
    if (params.criterion == CRITERION_GINI) { //инициализация для старта
        best_quality = 1e18; //относительно минимазации
    } else {
        best_quality = -1e18; //относительно максимизации
    }
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

            double g_left; //качество разделения левой части
            double g_right; // правой части

            if (params.criterion == CRITERION_GINI){ //определяем критерий
                g_left = counter_gini(y_left, left_count, params.num_classes);
                g_right = counter_gini( y_right, right_count, params.num_classes);
            } else {
                g_left = counter_entropy(y_left, left_count, params.num_classes);
                g_right = counter_entropy(y_right, right_count, params.num_classes);
            }

            double quality; //качество разделение

            if (params.criterion == CRITERION_GINI) {
                quality = ((double)left_count / n_objects) * g_left + //подсчет gini
                    ((double)right_count / n_objects) * g_right;
            } else {
                double parent_entropy = counter_entropy(y, n_objects, params.num_classes); //энтропия узла

                double children_entropy = ((double)left_count / n_objects) * g_left + // потомков
                    ((double)right_count / n_objects) * g_right;

                quality = parent_entropy - children_entropy; // коэф прироста информации
            }

            if ((params.criterion == CRITERION_GINI && quality < best_quality) ||
                (params.criterion == CRITERION_ENTROPY && quality > best_quality)){ // проверка лучше ли тек разбиение
                best_quality = quality;
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

Node_Tree *build_regression_tree(double **X, double *y, int n_objects,                          //построение дерева регресси
    int n_features, int depth, Params_Tree params){
    if (n_objects == 0) //отсутсвие объектов
        return NULL;

    if (depth >= params.max_depth || n_objects < params.min_objects_split){ //условие остановки
        return create_regression_leaf(y, n_objects);
    }

    int m = params.num_features_to_try; //случайные признаки

    if (m <= 0 || m > n_features)
        m = n_features;

    int *features = xmalloc(m * sizeof(int));

    choose_random_features(n_features, m, features); //выбор случайных признаков

    double best_mse = 1e18; //инициализация
    int best_feature = -1;
    double best_limit = 0.0;

    for (int fi = 0; fi < m; fi++) { //перебор случайных признаков
        int feature = features[fi];

        for (int i = 0; i < n_objects; i++) {

            double limit = X[i][feature];//перебор порогов

            int left_count = 0; //счетчики разделения
            int right_count = 0;

            for (int j = 0; j < n_objects; j++) { //первичное разделение для нахождение лучшего 
                if (X[j][feature] < limit)
                    left_count++;
                else
                    right_count++;
            }

            if (left_count == 0 || right_count == 0) //неудачное разделение
                continue; 

            double *y_left = xmalloc(left_count * sizeof(double));

            double *y_right = xmalloc(right_count * sizeof(double));

            int li = 0;
            int ri = 0;

            for (int j = 0; j < n_objects; j++) {

                if (X[j][feature] < limit) //повтороное разделение для значение на mse
                    y_left[li++] = y[j];
                else
                    y_right[ri++] = y[j];
            }

            double mse_left = counter_mse(y_left, left_count);
            double mse_right = counter_mse(y_right, right_count);

            double mse = ((double)left_count / n_objects) * mse_left + // общее mse
                ((double)right_count / n_objects) * mse_right;

            if (mse < best_mse) { //перезапись разделения

                best_mse = mse;
                best_feature = feature;
                best_limit = limit;
            }

            free(y_left);
            free(y_right);
        }
    }

    free(features);

    if (best_feature == -1)// разделение нет
        return create_regression_leaf(y, n_objects);

    int left_count = 0; //счетчики 
    int right_count = 0;

    for (int i = 0; i < n_objects; i++) { //разделение для нахождения количества в частяъ
        if (X[i][best_feature] < best_limit)
            left_count++;
        else
            right_count++;
    }

    double **X_left = xmalloc(left_count * sizeof(double*));
    double **X_right = xmalloc(right_count * sizeof(double*));

    double *y_left = xmalloc(left_count * sizeof(double));
    double *y_right = xmalloc(right_count * sizeof(double));

    int li = 0;
    int ri = 0;

    for (int i = 0; i < n_objects; i++) { //разделение
        if (X[i][best_feature] < best_limit) {
            X_left[li] = X[i];
            y_left[li] = y[i];
            li++;
        } else {
            X_right[ri] = X[i];
            y_right[ri] = y[i];
            ri++;
        }
    }
    Node_Tree *node = xcalloc(1, sizeof(Node_Tree)); //создание узла
    node->node_leaf = 0;
    node->feature_index = best_feature;
    node->limit = best_limit;

    node->left = build_regression_tree( X_left, y_left, left_count,
            n_features, depth + 1, params);

    node->right = build_regression_tree( X_right, y_right, right_count,
            n_features, depth + 1, params);

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

double predict_regression_tree(Node_Tree *root, const double *x){
    Node_Tree *cur = root;
    while (cur && !cur->node_leaf) {
        if (x[cur->feature_index] < cur->limit) cur = cur->left;
        else cur = cur->right;
    }
    return cur ? cur->predicted_value : 0.0;
}

void free_tree(Node_Tree *root) {        //освобождение дерева
    if (!root) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}

void print_tree(Node_Tree *node, int depth, Task_Type task_type){               //ASCII вызуализация
    if (!node)
        return;

    for (int i = 0; i < depth; i++) { //соблюдение отступов уровня
        printf("    ");
    }

    if (node->node_leaf) { //лист
        if (task_type == TASK_CLASSIFICATION) {
            printf("class = %d\n", node->predicted_class); //классификация
        } else {
            printf("value = %.3f\n", node->predicted_value); //регрессия
        }

        return;
    }

    printf("[X[%d] < %.3f]\n", node->feature_index, node->limit); //узел - условие

    for (int i = 0; i < depth; i++)
        printf("    ");

    printf("|--- yes:\n");

    print_tree(node->left, depth + 1, task_type);

    for (int i = 0; i < depth; i++)
        printf("    ");

    printf("|___ no:\n");

    print_tree(node->right, depth + 1, task_type);
}

void save_tree(FILE *f, Node_Tree *node, Task_Type task_type){                  //сохранение дерева в .txt
    if (!node)
        return;

    if (node->node_leaf) { //лист
        if (task_type == TASK_CLASSIFICATION) {
            fprintf(f, "L %d\n", node->predicted_class);
        } else {
            fprintf(f, "L %lf\n", node->predicted_value);
        }

        return;
    }

    fprintf(f, "N %d %lf\n", node->feature_index, node->limit);

    save_tree(f, node->left, task_type);

    save_tree(f, node->right, task_type);

    fprintf(f, "E\n");
}

static int global_node_id = 0;

static int write_dot_node(FILE *f, Node_Tree *node, Task_Type task_type){
    if (!node)
        return -1;

    int current_id = global_node_id++; //номер узла

    if (node->node_leaf) { //лист
        if (task_type == TASK_CLASSIFICATION) { //классификация
            fprintf(f,
                "node%d [shape=box, style=filled, fillcolor=lightblue, label=\"class = %d\"];\n",
                current_id,
                node->predicted_class);
        } else {
            fprintf(f,         //для регресии
                "node%d [shape=box, style=filled, fillcolor=lightgreen, label=\"value = %.3f\"];\n",
                current_id,
                node->predicted_value);
        }

        return current_id;
    }

    fprintf(f, //узел
        "node%d [shape=ellipse, style=filled, fillcolor=lightyellow, label=\"X[%d] < %.3f\"];\n",
        current_id,
        node->feature_index,
        node->limit);

    int left_id = write_dot_node(f, node->left, task_type);

    int right_id = write_dot_node(f, node->right, task_type);

    if (left_id != -1) {

        fprintf(f, "node%d -> node%d [label=\"yes\"];\n", current_id, left_id); //соединяем стрелкой
    }

    if (right_id != -1) {
        fprintf(f, "node%d -> node%d [label=\"no\"];\n", current_id, right_id); //соединяем стрелкой
    }

    return current_id;
}

void export_tree_to_png(Node_Tree *root, const char *dot_filename,                      //.png дерево
                        const char *png_filename, Task_Type task_type){
    FILE *f = fopen(dot_filename, "w");

    if (!f) {
        printf("Couldn't create dot file\n"); //отсутсвие файла
        return;
    }

    fprintf(f, "digraph Tree {\n");
    fprintf(f, "node [fontname=\"Arial\"];\n");

    global_node_id = 0;

    write_dot_node(f,root, task_type); //рекурсия
    fprintf(f, "}\n");
    fclose(f);

    char command[512]; //сборка команд

    snprintf(command, sizeof(command), "dot -Tpng %s -o %s",
              dot_filename, png_filename); // создание .png

    system(command);

    printf("Tree visualization saved to %s\n",
           png_filename);
}

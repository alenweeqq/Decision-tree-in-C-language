#include "dataset.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int count_columns_in_dataset_line(const char *string) {      //количество столбцов в датасете
    int column = 1;
    for (const char *symbol = string; *symbol; symbol++) {
        if (*symbol == ',') column++; // если встретилась в строке "," => новый столбец
    }
    return column;
}

Dataset *load_dataset(const char *filename) {                //загрузка датасета
    FILE *fp = fopen(filename, "r");
    if (!fp) emergency_exit("Failed to open the CSV file");

    char line[MAX_LEN_LINE]; //построчное храниие

    int n_object = 0; //инициализация количества строк (объектов) 
    int total_cols = -1; //стоблцов

    while (fgets(line, sizeof(line), fp)) { //чтение
        if (strlen(line) <= 1) continue; //пропуск пустых строк
        if (total_cols == -1) {
            total_cols = count_columns_in_dataset_line(line); //считаем количество стоблцов относительно 1 строки
        }                                                                       
        n_object++;
    }

    if (n_object == 0 || total_cols < 2) { //ошибка: отсутсвие строк или признков/ класса
        fclose(fp);
        emergency_exit("The CSV file is empty or invalid");
    }

    rewind(fp); // возращенеи в начало для загруски

    Dataset *ds = (Dataset *)xcalloc(1, sizeof(Dataset));
    ds->n_objects = n_object; //заполняем размеры
    ds->n_features = total_cols - 1; //оставляем класс
    ds->X = (double **)xmalloc(n_object * sizeof(double *)); //массив указателей строки / двумерный массив по объекту и признаку
    ds->y = (int *)xmalloc(n_object * sizeof(int)); //массив классов

    for (int i = 0; i < n_object; i++) {
        ds->X[i] = (double *)xmalloc(ds->n_features * sizeof(double));
    }

    int string = 0; //строка
    while (fgets(line, sizeof(line), fp)) {
        if (strlen(line) <= 1) continue; //пропускаем пустые строки

        char *token = strtok(line, ","); //разбиваем по значениям
        int col = 0;

        while (token) {
            if (col < ds->n_features) {
                ds->X[string][col] = atof(token); // записываем признаки в double типе
            } else if (col == ds->n_features) {
                ds->y[string] = atoi(token); //записываем класс в типе int
            } else {
                fclose(fp);
                emergency_exit("There are too many columns in the CSV row");
            }
            token = strtok(NULL, ",");//переходим к следущему элементу
            col++;
        }

        if (col != total_cols) {
            fclose(fp);
            emergency_exit("Inconsistent number of columns in CSV");
        }

        string++;
    }

    fclose(fp);

    ds->num_classes = max_el(ds->y, ds->n_objects) + 1; //находим максимальнй номер класса для нахождение количества классов
    return ds;
}

void free_dataset(Dataset *ds) {             //освобождение памяти датасета
    if (!ds) return; //NULL, защита от ошибки 
    for (int i = 0; i < ds->n_objects; i++) { //массив указателей
        free(ds->X[i]);
    }
    free(ds->X);
    free(ds->y);
    free(ds);
}

void split_dataset(const Dataset *src, double test_ratio,        //разделение датасета на train (обучение) и test (проверка)
                          Dataset **train_out, Dataset **test_out) {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) { //обрабатываем долю теста
        emergency_exit("test_ratio should be in (0, 1)");
    }

    int n = src->n_objects;
    int *indices = (int *)xmalloc(n * sizeof(int)); 
    for (int i = 0; i < n; i++) indices[i] = i; //массив индексов
    uniform_mixing(indices, n); //алгоритм Фишера–Йетса

    int test_n = (int)(n * test_ratio); //размеры теста
    if (test_n < 1) test_n = 1; //случай, когда берем 1 тест 
    int train_n = n - test_n; //остаток на обучение
    if (train_n < 1) emergency_exit("Not enough data after split"); //проверка, на случай есть на обучение не осталось

    Dataset *train = (Dataset *)xcalloc(1, sizeof(Dataset)); //создаем датасет обучение 
    Dataset *test = (Dataset *)xcalloc(1, sizeof(Dataset)); //датасет теста

    train->n_objects = train_n;
    train->n_features = src->n_features;
    train->num_classes = src->num_classes;
    train->X = (double **)xmalloc(train_n * sizeof(double *));
    train->y = (int *)xmalloc(train_n * sizeof(int));

    test->n_objects = test_n;
    test->n_features = src->n_features; //передаем данные
    test->num_classes = src->num_classes;
    test->X = (double **)xmalloc(test_n * sizeof(double *));// выделяем память (признаки)
    test->y = (int *)xmalloc(test_n * sizeof(int)); //выделяем память (класс)

    for (int i = 0; i < train_n; i++) {
        int idx = indices[i];
        train->X[i] = (double *)xmalloc(src->n_features * sizeof(double));
        for (int j = 0; j < src->n_features; j++) {
            train->X[i][j] = src->X[idx][j]; //копируем данные(признаки) для обучения
        }
        train->y[i] = src->y[idx];
    }

    for (int i = 0; i < test_n; i++) {
        int idx = indices[train_n + i];
        test->X[i] = (double *)xmalloc(src->n_features * sizeof(double));
        for (int j = 0; j < src->n_features; j++) {
            test->X[i][j] = src->X[idx][j]; //копируем данные(признаки) для теста
        }
        test->y[i] = src->y[idx];
    }

    free(indices);

    *train_out = train;
    *test_out = test;
}
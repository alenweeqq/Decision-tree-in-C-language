#include "dataset.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int count_columns_in_dataset_line(const char *string) {      //количество столбцов в датасете
    int column = 1; //инициализация

    for (const char *symbol = string; *symbol; symbol++) {
        if (*symbol == ',') {
            column++; // если встретилась в строке "," => новый столбец
        }
    }

    return column;
}

Dataset *load_dataset(const char *filename, Task_Type task_type) {                //загрузка датасета
    FILE *fp = fopen(filename, "r");

    if (!fp) {
        emergency_exit("Failed to open the CSV file"); //ошибка открытия
    }

    char line[MAX_LEN_LINE]; //построчное храниие

    fgets(line, sizeof(line), fp); // пропуск заголовков

    int n_object = 0; //инициализация количества строк (объектов)
    int total_cols = -1; //стоблцов

    while (fgets(line, sizeof(line), fp)) { //чтение

        if (strlen(line) <= 1) {
            continue; //пропуск пустых строк
        }

        if (total_cols == -1) {
            total_cols = count_columns_in_dataset_line(line); //считаем количество стоблцов относительно 1 строки
        }

        n_object++;
    }

    if (n_object == 0 || total_cols < 2) { //ошибка: отсутсвие строк или признков
        fclose(fp);
        emergency_exit("The CSV file is empty or invalid");
    }

    rewind(fp); // возращенеи в начало для загруски
    fgets(line, sizeof(line), fp); // снова пропускаем заголовок

    Dataset *ds = (Dataset *)xcalloc(1, sizeof(Dataset)); 

    ds->n_objects = n_object; //заполняем размеры 
    ds->n_features = total_cols - 1; //оставляем класс/ значения

    ds->X = (double **)xmalloc(n_object * sizeof(double *)); //массив указателей строки / двумерный массив по объекту и признаку

    if (task_type == TASK_CLASSIFICATION) { //классификация или регрессия
        ds->y_class = xmalloc(n_object * sizeof(int));
    } else {
        ds->y_reg = xmalloc(n_object * sizeof(double));
    }

    for (int i = 0; i < n_object; i++) {
        ds->X[i] = (double *)xmalloc(ds->n_features * sizeof(double)); //выделенияе памяти для признаков
    }

    int string = 0; //строка

    while (fgets(line, sizeof(line), fp)) {

        if (strlen(line) <= 1) {
            continue; //пропускаем пустые строки
        }

        char *token = strtok(line, ","); //разбиваем по значениям
        int col = 0;//столбец

        while (token) {
            token[strcspn(token, "\r\n")] = '\0'; //удаление переноса
            if (col < ds->n_features) { //признак
                ds->X[string][col] = atof(token); // записываем признаки в double типе
            } else if (col == ds->n_features) { //тип классификации или регрессия
                if (task_type == TASK_CLASSIFICATION) {
                    ds->y_class[string] = atoi(token);
                } else {
                    ds->y_reg[string] = atof(token);
                }
            } else {
                fclose(fp);
                emergency_exit("There are too many columns in the CSV row"); //изботок столбцов
            }

            token = strtok(NULL, ",");//переходим к следущему элементу
            col++;
        }

        if (col != total_cols) {

            fclose(fp);
            emergency_exit("Inconsistent number of columns in CSV"); //некорректное число столбцов
        }

        string++;
    }

    fclose(fp);

    if (task_type == TASK_CLASSIFICATION) { //для классификации
        ds->num_classes = max_el(ds->y_class, ds->n_objects) + 1; //отпеределяем кол классов
    }
    return ds;
}

void free_dataset(Dataset *ds) {             //освобождение памяти датасета

    if (!ds) {
        return; //NULL, защита от ошибки
    }

    for (int i = 0; i < ds->n_objects; i++) { //массив указателей
        free(ds->X[i]);
    }

    free(ds->X); //признаки
    free(ds->y_class); //классы
    free(ds->y_reg); //значения регрессии
    free(ds); //датасет
}

void split_dataset(const Dataset *src, double test_ratio,        //разделение датасета на train (обучение) и test (проверка)
                   Dataset **train_out, Dataset **test_out) {

    if (test_ratio <= 0.0 || test_ratio >= 1.0) { //обрабатываем долю теста
        emergency_exit("test_ratio should be in (0, 1)");
    }

    int n = src->n_objects; //количество объектов

    int *indices = (int *)xmalloc(n * sizeof(int)); //индексы

    for (int i = 0; i < n; i++) {
        indices[i] = i; //массив индексов
    }

    uniform_mixing(indices, n); //алгоритм Фишера–Йетса для равномерного перемешивания

    int test_n = (int)(n * test_ratio); //размеры теста

    if (test_n < 1) {
        test_n = 1; //случай, когда берем 1 тест
    }

    int train_n = n - test_n; //остаток на обучение

    if (train_n < 1) {
        emergency_exit("Not enough data after split"); //проверка что на обучение есть данные
    }

    Dataset *train = (Dataset *)xcalloc(1, sizeof(Dataset)); //создаем датасет обучение
    Dataset *test = (Dataset *)xcalloc(1, sizeof(Dataset)); //датасет теста

    train->n_objects = train_n; //копируем параметры
    train->n_features = src->n_features;
    train->num_classes = src->num_classes;

    test->n_objects = test_n;
    test->n_features = src->n_features;
    test->num_classes = src->num_classes;

    train->X = (double **)xmalloc(train_n * sizeof(double *)); //признаки
    test->X = (double **)xmalloc(test_n * sizeof(double *));

    if (src->y_class) { //классификация
        train->y_class = xmalloc(train_n * sizeof(int));
        test->y_class = xmalloc(test_n * sizeof(int));
    }

    if (src->y_reg) { //регрессия
        train->y_reg = xmalloc(train_n * sizeof(double));
        test->y_reg = xmalloc(test_n * sizeof(double));
    }

    for (int i = 0; i < train_n; i++) { //обучение
        int idx = indices[i];

        train->X[i] = (double *)xmalloc(src->n_features * sizeof(double));
        for (int j = 0; j < src->n_features; j++) {
            train->X[i][j] = src->X[idx][j]; //копируем данные(признаки) для обучения
        }

        if (src->y_class) { //классы
            train->y_class[i] = src->y_class[idx];
        }

        if (src->y_reg) { //значения
            train->y_reg[i] = src->y_reg[idx];
        }
    }

    for (int i = 0; i < test_n; i++) { //тесты
        int idx = indices[train_n + i];
        test->X[i] = (double *)xmalloc(src->n_features * sizeof(double));
        for (int j = 0; j < src->n_features; j++) {
            test->X[i][j] = src->X[idx][j]; //копируем данные(признаки) для теста
        }
        if (src->y_class) {//классы
            test->y_class[i] = src->y_class[idx];
        }
        if (src->y_reg) { //значения
            test->y_reg[i] = src->y_reg[idx];
        }
    }

    free(indices);

    *train_out = train;
    *test_out = test;
}

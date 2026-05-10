#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dataset.h"
#include "forest.h"
#include "utils.h"
#include "tree.h"

static void print_usage(const char *prog) {                 //инструкция запуска
    printf("Launch Instructions:\n");
    printf("  %s <csv file(dataset)> <number of trees> <maximum depth>"
        "<minimum of objects to divide> <number of random selection of features> <proportion of test>\n", prog);
    printf("\nExample:\n");
    printf("  %s dataset.csv 15 6 2 2 0.2\n", prog);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        print_usage(argv[0]); //выводим инструкцию
        return 1;
    }

    srand((unsigned int)time(NULL)); //генератор случайных чисел

    const char *csv_file = argv[1];  // принимаем параметры
    int n_trees = atoi(argv[2]);
    int max_depth = atoi(argv[3]); 
    int min_objects_split = atoi(argv[4]);
    int num_features_to_try = atoi(argv[5]);
    double test_ratio = atof(argv[6]);

    if (n_trees <= 0 || max_depth <= 0 || min_objects_split <= 0) {
        emergency_exit("Incorrect parameters");
    }

    Dataset *full = load_dataset(csv_file); //загрузка датасета
    Dataset *train = NULL;
    Dataset *test = NULL;

    split_dataset(full, test_ratio, &train, &test); //делим датасет на обучение и тест

    Params_Tree params; //фиксируем параметры
    params.max_depth = max_depth;
    params.min_objects_split = min_objects_split;
    params.num_features_to_try = num_features_to_try;
    params.num_classes = train->num_classes;

    Forest *forest = train_forest(train->X, train->y, train->n_objects, //создаем лес и обучаем
                                  train->n_features, n_trees, params);

    double train_acc = accuracy_forest(forest, train); //точность модели на обучении
    double test_acc = accuracy_forest(forest, test); //точность модели на тесте

    printf("Uploaded data set:\n");
    printf("  Objects: %d\n", full->n_objects);
    printf("  Features: %d\n", full->n_features);
    printf("  Classes: %d\n", full->num_classes);
    printf("\nSplit:\n");
    printf("  Train: %d\n", train->n_objects);
    printf("  Test: %d\n", test->n_objects);
    printf("\nModel:\n");
    printf("  Trees: %d\n", n_trees);
    printf("  Max depth: %d\n", max_depth);
    printf("  Min objects split: %d\n", min_objects_split);
    printf("  Random features per node: %d\n", num_features_to_try);
    printf("\nAccuracy:\n");
    printf("  Train accuracy: %.4f\n", train_acc);
    printf("  Test accuracy:  %.4f\n", test_acc);

    printf("\nPredictions on the first 5 tests:\n");
    int limit = test->n_objects < 5 ? test->n_objects : 5;
    for (int i = 0; i < limit; i++) {
        int pred = predict_forest(forest, test->X[i]);
        printf("  Object %d: true=%d pred=%d\n", i, test->y[i], pred);
    }

    printf("\n--- Visualization of 1 tree---\n");
    print_tree(forest->trees[0], 0);

    save_forest(forest, "model.txt");
    printf("\nModel saved to model.txt\n");

    free_forest(forest);
    free_dataset(train);
    free_dataset(test);
    free_dataset(full);

    return 0;
}
// .\random_forest.exe dataset.csv 10 8 4 2 0.4 
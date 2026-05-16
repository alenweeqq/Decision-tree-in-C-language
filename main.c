#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "dataset.h"
#include "forest.h"
#include "utils.h"
#include "tree.h"

static void print_usage(const char *prog) { //инструкция запуска
    printf("Launch Instructions:\n");
    printf("  %s <csv file(dataset)> <number of trees> <maximum depth> <minimum objects split> "
           "<random features> <test ratio> <task type> [criterion]\n", prog);

    printf("\nTask type:\n");
    printf("  classification\n");
    printf("  regression\n");

    printf("\nCriterion:\n");
    printf("  gini\n");
    printf("  entropy\n");

    printf("\nExample:\n");
    printf("  %s dataset.csv 15 6 2 2 0.2 classification gini\n", prog);
    printf("  %s dataset.csv 15 6 2 2 0.2 regression\n", prog);
}

int main(int argc, char **argv){
    if (argc != 8 && argc != 9){
        print_usage(argv[0]);
        return 1;
    }
    srand((unsigned int)time(NULL)); //генератор случайных чисел
    const char *csv_file = argv[1];
    int n_trees = atoi(argv[2]); //инициализация
    int max_depth = atoi(argv[3]);
    int min_objects_split = atoi(argv[4]);
    int num_features_to_try = atoi(argv[5]);
    double test_ratio = atof(argv[6]);
    const char *task_name = argv[7];

    Params_Tree params;
    if (strcmp(task_name, "classification") == 0) { //опеределения типа задачи
        params.task_type = TASK_CLASSIFICATION;
    } else if (strcmp(task_name, "regression") == 0) {
        params.task_type = TASK_REGRESSION;
    } else {
        emergency_exit("Unknown task type");
    }

    if (params.task_type == TASK_CLASSIFICATION) {
        if (argc != 9) {
            emergency_exit("Criterion required for classification");
        }

        const char *criterion_name = argv[8];
        if (strcmp(criterion_name, "gini") == 0) {      //определение коэфицента
            params.criterion = CRITERION_GINI;

        } else if (strcmp(criterion_name, "entropy") == 0) {
            params.criterion = CRITERION_ENTROPY;
        } else {
            emergency_exit("Unknown criterion");
        }
    }

    if(n_trees <= 0 || max_depth <= 0 || min_objects_split <= 0){ //некорректные параметры
        emergency_exit("Incorrect parameters");
    }

    Dataset *full = load_dataset(csv_file, params.task_type);

    Dataset *train = NULL;
    Dataset *test = NULL;

    split_dataset(full, test_ratio, &train, &test); //разделение датасета
    params.max_depth = max_depth; //инициализация
    params.min_objects_split = min_objects_split;
    params.num_features_to_try = num_features_to_try;

    params.num_classes = train->num_classes;

    Forest *forest = NULL;

    if(params.task_type == TASK_CLASSIFICATION){ //создание леса по типу задачи
        forest = train_forest( train->X, train->y_class, train->n_objects,
                train->n_features, n_trees, params);
    } else {
        forest = train_regression_forest( train->X, train->y_reg, train->n_objects,
                train->n_features, n_trees, params);
    }

    printf("\n--- MODEL INFORMATION ---\n");

    print_model_info(forest);
   if (params.task_type == TASK_CLASSIFICATION){
        double train_acc = accuracy_forest(forest, train); //точность леса классификации
        double test_acc = accuracy_forest(forest, test);
        printf("Train accuracy: %.4f\n", train_acc);
        printf("Test accuracy: %.4f\n", test_acc);
    } else {
        double train_mse = mse_forest(forest, train); //точность леса регрессии
        double test_mse = mse_forest(forest, test);
        printf("Train MSE: %.4f\n", train_mse);
        printf("Test MSE: %.4f\n", test_mse);
    }

    printf("\n--- TREE VISUALIZATION ---\n");

    print_tree(forest->trees[0], 0, params.task_type);

    export_tree_to_png( forest->trees[0], "tree.dot", "tree.png", params.task_type);

    printf("\nTree saved:\n");
    printf("  tree.dot\n");
    printf("  tree.png\n");

    save_forest(forest, "model.txt");

    printf("\nModel saved to model.txt\n");

    free_forest(forest);
    free_dataset(train);
    free_dataset(test);
    free_dataset(full);

    return 0;
}
// .\random_forest.exe dataset.csv 15 6 2 2 0.2 classification gini
// .\random_forest.exe dataset.csv 15 6 2 2 0.2 regression

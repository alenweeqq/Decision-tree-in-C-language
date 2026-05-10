#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void emergency_exit(const char *msg) {          //аварийный выход
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

void *xmalloc(size_t size) {                    //выделение памяти(malloc) с безопасным выходом в случае ошибки
    void *ptr = malloc(size);
    if (!ptr) emergency_exit("Couldn't allocate memory");
    return ptr;
}

void *xcalloc(size_t count, size_t size) {       //выделение памяти(calloc) с безопасным выходом в случае ошибки
    void *ptr = calloc(count, size);
    if (!ptr) emergency_exit("Couldn't allocate memory");
    return ptr;
}

void uniform_mixing(int *arr, int n) {          //равномерно перемешивание
    for (int i = n - 1; i > 0; i--) {      //алгоритм Фишера–Йетса
        int j = rand() % (i + 1); //рандомный индекс от 0 до i
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

int max_el(const int *arr, int n) {             //взятие масимального элемента
    int mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > mx) mx = arr[i];
    }
    return mx;
}
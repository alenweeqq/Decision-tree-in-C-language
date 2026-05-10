#ifndef UTILS_H
#define UTILS_H

#include <stddef.h> 

void emergency_exit(const char *msg); //аварийный выход

void *xmalloc(size_t size); //выделение памяти(malloc) с безопасным выходом в случае ошибки

void *xcalloc(size_t count, size_t size);  //выделение памяти(calloc) с безопасным выходом в случае ошибки

void uniform_mixing(int *arr, int n); //равномерно перемешивание

int max_el(const int *arr, int n); //взятие масимального элемента

#endif
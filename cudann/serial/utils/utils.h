#ifndef UTILS_H
#define UTILS_H
#include <functional>
#include <iostream>

void where_double(double* out, std::function<const double(const double, const double)> cond, const double *x, const double *y, const int size);

#endif
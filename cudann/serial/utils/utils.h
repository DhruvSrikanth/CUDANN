#ifndef UTILS_H
#define UTILS_H
#include <functional>
#include <iostream>

void where_double(double* out, std::function<const double(const double, const double)> cond, const double *x, const double *y, const int size);
void pmf_to_class_pred(Tensor *pmf, const int *argmax);
double compare_one_hots(Tensor *y_hat, Tensor *y);

#endif
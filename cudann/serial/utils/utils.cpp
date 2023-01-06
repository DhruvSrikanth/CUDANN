#include <functional>
#include <iostream>
#include "tensor.h"

void where_double(double* out, std::function<const double(const double, const double)> cond, const double *x, const double *y, const int size) {
    for (int i = 0; i < size; i++) {
        if (cond(x[i], y[i])) {
            out[i] = x[i];
        } else {
            out[i] = y[i];
        }
    }
}

void pmf_to_class_pred(Tensor *pmf, const int *argmax) {
    for (int b = 0; b < pmf->batch_size; b++) {
        for (int f = 0; f < pmf->n_features; f++) {
            if (f == argmax[b]) {
                pmf->data[b * pmf->n_features + f] = 1.0;
            } else {
                pmf->data[b * pmf->n_features + f] = 0.0;
            }
        }
    }
}

double compare_one_hots(Tensor *y_hat, Tensor *y) {
    int correct = 0;
    for (int b = 0; b < y_hat->batch_size; b++) {
        int equal = 0;
        for (int f = 0; f < y_hat->n_features; f++) {
            if (y_hat->data[b * y_hat->n_features + f] == y->data[b * y->n_features + f]) {
                equal++;
            }
        }
        if (equal == y_hat->n_features) {
            correct++;
        }
    }

    return (double) correct / (double) y_hat->batch_size;
}
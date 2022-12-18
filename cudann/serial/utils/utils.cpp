#include <functional>
#include <iostream>

void where_double(double* out, std::function<const double(const double, const double)> cond, const double *x, const double *y, const int size) {
    for (int i = 0; i < size; i++) {
        if (cond(x[i], y[i])) {
            out[i] = x[i];
        } else {
            out[i] = y[i];
        }
    }
}
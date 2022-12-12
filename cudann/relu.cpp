#include <stdio.h>
#include <iostream>
#include <string.h>
#include "relu.h"

void ReLU::show() {
    std::printf("%s\n", this->name);
}

double* ReLU::forward(const double *input) {
    const int size = sizeof(input) / sizeof(input[0]);

    // Copy input to x
    if (this->first_call) {
        this->x = new double[size];
        this->fx = new double[size];
        this->dfx = new double[size];
        this->first_call = false;
    }

    memcpy(this->x, input, size * sizeof(double));

    // Compute relu activation
    relu_activation(size, this->x, this->fx);

    // Return output
    return this->fx;
}

// Backward call
double* ReLU::backward(const double *upstream_grad) {
    const int size = sizeof(upstream_grad) / sizeof(upstream_grad[0]);

    // Copy upstream gradient to dfx
    memcpy(this->dfx, upstream_grad, size * sizeof(double));

    // Compute relu gradient
    relu_gradient(size, this->x, this->dfx);

    // Return output
    return this->dfx;
}

// ReLU activation - y = max(0, x)
void relu_activation(const int size, const double *x, double *fx) {
    // Perform ReLU activation
    for (int i = 0; i < size; i++) {
        if (x[i] > 0) {
            fx[i] = x[i];
        } else {
            fx[i] = 0;
        }
    }
}

// ReLU gradient - y = fx > 0
void relu_gradient(const int size, const double *x, double *dfx) {
    // Perform ReLU gradient
    for (int i = 0; i < size; i++) {
        if (x[i] > 0) {
            dfx[i] = dfx[i] * 1;
        } else {
            dfx[i] = dfx[i] * 0;
        }
    }
}

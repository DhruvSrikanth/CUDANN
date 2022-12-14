#include <stdio.h>
#include <iostream>
#include <string.h>
#include "relu.h"


// Initialize the ReLU Layer
ReLU::ReLU(const int n_features, const std::string name="Sigmoid") {
    // Set the number of features
    this->n_features = n_features;

    // Set layer name
    this->name = name;

    // Initialize input, output, input gradient and output gradient
    this->x = (double*) malloc(n_features * sizeof(double));
    this->fx = (double*) malloc(n_features * sizeof(double));
    this->dfx = (double*) malloc(n_features * sizeof(double));
}

// Destructor
ReLU::~ReLU() {
    free(this->x);
    free(this->fx);
    free(this->dfx);
}

// Print layer
void ReLU::show() {
    std::printf("%s: [%d]\n", this->name, this->n_features);
}

double* ReLU::forward(const double *input) {
    const int size = sizeof(input) / sizeof(input[0]);

    // Reallocate memory for input, output and input gradient
    this->x = (double*) realloc(this->x, size * sizeof(double));
    this->fx = (double*) realloc(this->fx, size * sizeof(double));
    memcpy(this->x, input, size * sizeof(double));

    // Compute relu activation
    relu_activation_batch(this->x, this->fx, size, this->n_features);

    // Return output
    return this->fx;
}

// Backward call
double* ReLU::backward(const double *upstream_grad) {
    const int size = sizeof(upstream_grad) / sizeof(upstream_grad[0]);

    // Copy upstream gradient to dfx
    this->dfx = (double*) realloc(this->dfx, size * sizeof(double));
    memcpy(this->dfx, upstream_grad, size * sizeof(double));

    // Compute relu gradient
    relu_gradient_batch(this->x, this->dfx, size, this->n_features);

    // Return output
    return this->dfx;
}

// ReLU activation on a batch - y (b, n_features) = max(0, x) (b, n_features)
void relu_activation_batch(const double *x, double *fx, const int size, const int n_features) {
    // Perform ReLU activation on each batch
    const int n_batches = size / n_features;
    for (int b = 0; b < n_batches; b++) {
        relu_activation(b, x, fx, n_features);
    }
    
}

// ReLU activation - y = max(0, x)
void relu_activation(const int b, const double *x, double *fx, const int n_features) {
    // Perform ReLU activation
    for (int i = 0; i < n_features; i++) {
        if (x[b*n_features + i] > 0) {
            fx[b*n_features + i] = x[b*n_features + i];
        } else {
            fx[b*n_features + i] = 0;
        }
    }
}

// ReLU gradient on a batch - y (b, n_features) = max(0, x) (b, n_features)
void relu_gradient_batch(const double *x, double *dfx, const int size, const int n_features) {
    // Perform ReLU gradient on each batch
    const int n_batches = size / n_features;
    for (int b = 0; b < n_batches; b++) {
        relu_gradient(b, x, dfx, n_features);
    }   
}

// ReLU gradient - y = fx > 0
void relu_gradient(const int b, const double *x, double *dfx, const int n_features) {
    // Perform ReLU gradient
    for (int i = 0; i < n_features; i++) {
        if (x[b*n_features + i] > 0) {
            dfx[b*n_features + i] = dfx[b*n_features + i] * 1;
        } else {
            dfx[b*n_features + i] = dfx[b*n_features + i] * 0;
        }
    }
}

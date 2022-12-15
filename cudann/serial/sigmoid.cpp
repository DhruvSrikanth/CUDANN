#include <iostream>
#include <string.h> 
#include <cmath>
#include "sigmoid.h"

// Initialize the Sigmoid Layer
Sigmoid::Sigmoid(const int n_features, const std::string name="Sigmoid") {
    // Set layer name
    this->name = name;

    // Initialize input, output, input gradient and output gradient
    this->n_features = n_features;
    this->x = (double*) malloc(n_features * sizeof(double));
    this->fx = (double*) malloc(n_features * sizeof(double));
    this->dfx = (double*) malloc(n_features * sizeof(double));
}

// Destructor
Sigmoid::~Sigmoid() {
    free(this->x);
    free(this->fx);
    free(this->dfx);
}

// Print layer name
void Sigmoid::show() {
    std::printf("%s: [%d]\n", this->name, this->n_features);
}

// Forward call
double* Sigmoid::forward(const double *input) {
    const size_t size = sizeof(input) / sizeof(input[0]);

    // Reallocate memory for input, output and input gradient
    this->x = (double*) realloc(this->x, size * sizeof(double));
    this->fx = (double*) realloc(this->fx, size * sizeof(double));
    memcpy(this->x, input, size * sizeof(double));

    // Compute sigmoid transformation
    sigmoid_activation_batch(this->x, this->fx, size, this->n_features);

    // Return output
    return this->fx;
}

// Backward call
double* Sigmoid::backward(const double *upstream_grad) {
    const size_t size = sizeof(upstream_grad) / sizeof(upstream_grad[0]);

    // Copy upstream gradient to dfx
    this->dfx = (double*) realloc(this->dfx, size * sizeof(double));
    memcpy(this->dfx, upstream_grad, size * sizeof(double));

    // Compute sigmoid gradient
    sigmoid_gradient_batch(this->fx, this->dfx, size, this->n_features);

    // Return output
    return this->dfx;
}
    
// Sigmoid activation on batch - y(b, n_features) = 1 / (1 + e^-x) (b, n_features)
void sigmoid_activation_batch(const double *x, double *fx, const int size, const int n_features) {
    // Perform sigmoid activation on each batch
    const int n_batches = size / n_features;
    for (int b = 0; b < n_batches; b++) {
        sigmoid_activation(b, x, fx, n_features);
    }
}

// Sigmoid activation - y = 1 / (1 + e^-x)
void sigmoid_activation(const int b, const double *x, double *fx, const int n_features) {
    // Compute sigmoid activation
    for (int i = 0; i < n_features; i++) {
        fx[b * n_features + i] = 1 / (1 + exp(-x[b * n_features + i]));
    }
}

// Sigmoid gradient on batch - y(b, n_features) = fx * (1 - fx) (b, n_features)
void sigmoid_gradient_batch(const double *fx, double *dfx, const int size, const int n_features) {
    // Compute sigmoid gradient
    const int n_batches = size / n_features;
    for (int b = 0; b < n_batches; b++) {
        sigmoid_gradient(b, fx, dfx, n_features);
    }
}

// Sigmoid gradient - y = fx * (1 - fx)
void sigmoid_gradient(const int b, const double *fx, double *dfx, const int n_features) {
    // Compute sigmoid gradient
    for (int i = 0; i < n_features; i++) {
        dfx[b * n_features + i] *= fx[b * n_features + i] * (1 - fx[b * n_features + i]);
    }
}
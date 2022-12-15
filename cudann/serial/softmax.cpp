#include <iostream>
#include <string.h> 
#include <cmath>
#include "softmax.h"

// Initialize the SigmSoftmaxoid Layer
Softmax::Softmax(const int n_classes, const std::string name="Softmax") {
    // Set layer name
    this->name = name;

    // Initialize input, output, input gradient and output gradient
    this->n_classes = n_classes;
    this->x = (double*) malloc(n_classes * sizeof(double));
    this->fx = (double*) malloc(n_classes * sizeof(double));
    this->dfx = (double*) malloc(n_classes * sizeof(double));
}

// Destructor
Softmax::~Softmax() {
    free(this->x);
    free(this->fx);
    free(this->dfx);
}

// Print layer name
void Softmax::show() {
    std::printf("%s: [%d]\n", this->name, this->n_classes);
}

// Forward call
double* Softmax::forward(const double *input) {
    const size_t size = sizeof(input) / sizeof(input[0]);

    // Reallocate memory for input, output and input gradient
    this->x = (double*) realloc(this->x, size * sizeof(double));
    this->fx = (double*) realloc(this->fx, size * sizeof(double));
    memcpy(this->x, input, size * sizeof(double));

    // Compute softmax transformation
    softmax_activation_batch(this->x, this->fx, size, this->n_classes);

    // Return output
    return this->fx;
}

// Backward call
double* Softmax::backward(const double *upstream_grad) {
    const size_t size = sizeof(fx) / sizeof(fx[0]);

    // Copy upstream gradient to dfx
    this->dfx = (double*) realloc(this->dfx, size * sizeof(double));
    memcpy(this->dfx, this->fx, size * sizeof(double));

    // Compute softmax gradient
    softmax_gradient_batch(upstream_grad, this->fx, this->dfx, size, this->n_classes);

    // Return output
    return this->dfx;
}
    
// Softmax activation on batch - y(b, n_classes) = e^(x(b, n_classes)) / sum(e^(x(b, n_classes))) (b, n_classes)
void softmax_activation_batch(const double *x, double *fx, const int size, const int n_classes) {
    // Perform softmax activation on each batch
    const int n_batches = size / n_classes;
    for (int b = 0; b < n_batches; b++) {
        softmax_activation(b, x, fx, n_classes);
    }
}

// Softmax activation - y = e^(x) / sum(e^(x))
void softmax_activation(const int b, const double *x, double *fx, const int n_classes) {
    // Compute softmax activation
    double exp_sum = 0;
    for (int i = 0; i < n_classes; i++) {
        fx[b * n_classes + i] = exp(x[b * n_classes + i]);
        exp_sum += fx[b * n_classes + i];
    }
    for (int i = 0; i < n_classes; i++) {
        fx[b * n_classes + i] /= exp_sum;
    }
}

// Softmax gradient on batch - y(b, n_classes) = fx(b, n_classes) * (upstream_grad(b, n_classes) - fx(b, n_classes) * upstream_grad(b, n_classes)) (b, n_classes)
void softmax_gradient_batch(const double* upstream_grad, const double *fx, double *dfx, const int size, const int n_classes) {
    // Compute softmax gradient
    const int n_batches = size / n_classes;
    for (int b = 0; b < n_batches; b++) {
        sigmoid_gradient(b, upstream_grad, fx, dfx, n_classes);
    }
}

// Softmax gradient - y = fx * (upstream_grad - fx * upstream_grad)
void sigmoid_gradient(const int b, const double* upstream_grad, const double *fx, double *dfx, const int n_classes) {
    // Compute softmax gradient
    for (int i = 0; i < n_classes; i++) {
        dfx[b * n_classes + i] *= (upstream_grad[b * n_classes + i] - (upstream_grad[b * n_classes + i] * fx[b * n_classes + i]));
    }
}
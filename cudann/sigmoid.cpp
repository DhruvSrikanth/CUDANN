#include <iostream>
#include <string.h> 
#include <cmath>
#include "sigmoid.h"

        // Initialize the Sigmoid Layer
Sigmoid::Sigmoid(const std::string name="Sigmoid") {
    // Set layer name
    this->name = name;

    // Initialize input, output, input gradient and output gradient
    this->x = NULL;
    this->fx = NULL;
    this->dfx = NULL;
}

// Print layer name
void Sigmoid::show() {
    std::printf("%s\n", this->name);
}

// Forward call
double* Sigmoid::forward(const double *input) {
    const int size = sizeof(input) / sizeof(input[0]);

    // Copy input to x
    if (this->first_call) {
        this->x = new double[size];
        this->fx = new double[size];
        this->dfx = new double[size];
        this->first_call = false;
    }

    memcpy(this->x, input, size * sizeof(double));

    // Compute sigmoid transformation
    sigmoid_transformation(size, this->x, this->fx);

    // Return output
    return this->fx;
}

// Backward call
double* Sigmoid::backward(const double *upstream_grad) {
    const int size = sizeof(upstream_grad) / sizeof(upstream_grad[0]);

    // Copy upstream gradient to dfx
    memcpy(this->dfx, upstream_grad, size * sizeof(double));

    // Compute sigmoid gradient
    sigmoid_gradient(size, this->fx, this->dfx);

    // Return output
    return this->dfx;
}
    
// Sigmoid transformation - y = 1 / (1 + e^-x)
void sigmoid_transformation(const int size, const double *x, double *fx) {
    // Perform sigmoid transformation
    for (int i = 0; i < size; i++) {
        fx[i] = 1 / (1 + std::exp(-x[i]));
    }
}

// Sigmoid gradient - y = fx * (1 - fx)
void sigmoid_gradient(const int size, const double *fx, double *dfx) {
    // Compute sigmoid gradient
    for (int i = 0; i < size; i++) {
        dfx[i] *= fx[i] * (1 - fx[i]);
    }
}
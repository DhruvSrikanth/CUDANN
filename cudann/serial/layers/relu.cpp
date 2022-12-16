#include "relu.h"
#include "../utils/tensor.h"
#include <cstring>

// Initialize the ReLU Layer
ReLU::ReLU(const int n_features, const std::string name) {
    // Set the number of features
    this->n_features = n_features;

    // Set layer name
    this->name = name;

    // Initialize input, output, input gradient and output gradient
    this->x = (tensor*) malloc(sizeof(tensor));
    this->fx = (tensor*) malloc(sizeof(tensor));
    this->dfx = (tensor*) malloc(sizeof(tensor));
    
}
    

// Destructor
ReLU::~ReLU() {
    free(this->x);
    free(this->fx);
    free(this->dfx);
}

// Print layer
void ReLU::show() {
    std::printf("%s: [%d]\n", this->name.c_str(), this->n_features);
}

tensor* ReLU::forward(const tensor *input) {
    const int size = input->n_batches * this->n_features;

    // Reallocate memory for input, output and input gradient
    tensor x = {input->n_batches, (double*) realloc(this->x->data, size * sizeof(double))};
    tensor fx = {input->n_batches, (double*) realloc(this->fx->data, size * sizeof(double))};
    this->x = &x;
    this->fx = &fx;

    memcpy(this->x->data, input->data, size * sizeof(double));

    // Compute relu activation
    relu_activation_batch(this->x->data, this->fx->data, size, this->n_features);

    // Return output
    return this->fx;
}

// Backward call
tensor* ReLU::backward(const tensor *upstream_grad) {
    const int size = upstream_grad->n_batches * this->n_features;

    // Copy upstream gradient to dfx
    this->dfx->n_batches = upstream_grad->n_batches;
    this->dfx->data = (double*) realloc(this->dfx->data, size * sizeof(double));
    memcpy(this->dfx->data, upstream_grad->data, size * sizeof(double));

    // Compute relu gradient
    relu_gradient_batch(this->x->data, this->dfx->data, size, this->n_features);

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

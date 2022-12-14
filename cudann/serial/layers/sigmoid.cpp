#include "sigmoid.h"
#include <cmath>
#include <cstring>

// Initialize the Sigmoid Layer
Sigmoid::Sigmoid(const int n_features, const std::string name) {
    // Set layer name
    this->name = name;

    this->n_features = n_features;
    
    // Initialize input, output, input gradient and output gradient
    this->x = NULL;
    this->fx = NULL;
    this->dfx = NULL;
}

// Destructor
Sigmoid::~Sigmoid() {
    // Free memory
    if (this->x != NULL) {
        free_tensor(this->x);
    }
    if (this->fx != NULL) {
        free_tensor(this->fx);
    }
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
}

// Print layer name
void Sigmoid::show() {
    std::printf("%s: [%d]\n", this->name.c_str(), this->n_features);
}

// Get layer type
std::string Sigmoid::get_type() {
    return this->type;
}

// Forward call
Tensor* Sigmoid::forward(const Tensor *input) {
    const int size = input->batch_size * this->n_features;

    // Reallocate memory for input, output and input gradient
    if (this->x != NULL) {
        free_tensor(this->x);
    }
    this->x = (Tensor*) malloc(sizeof(Tensor));
    // Copy input to x
    copy_tensor(this->x, input);

    if (this->fx != NULL) {
        free_tensor(this->fx);
    }
    this->fx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->fx, input->batch_size, this->n_features);

    // Compute sigmoid transformation
    sigmoid_activation_batch(this->x->data, this->fx->data, size, this->n_features);

    // Return output
    return this->fx;
}

// Backward call
Tensor* Sigmoid::backward(const Tensor *upstream_grad) {
    const int size = upstream_grad->batch_size * this->n_features;

    // Copy upstream gradient to dfx
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
    this->dfx = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->dfx, upstream_grad);

    // Compute sigmoid gradient
    sigmoid_gradient_batch(this->fx->data, this->dfx->data, size, this->n_features);

    // Return output
    return this->dfx;
}
    
// Sigmoid activation on batch - y(b, n_features) = 1 / (1 + e^-x) (b, n_features)
void sigmoid_activation_batch(const double *x, double *fx, const int size, const int n_features) {
    // Perform sigmoid activation on each batch
    const int batch_size = size / n_features;
    for (int b = 0; b < batch_size; b++) {
        sigmoid_activation(b, x, fx, n_features);
    }
}

// Sigmoid activation - y = 1 / (1 + e^-x)
void sigmoid_activation(const int b, const double *x, double *fx, const int n_features) {
    // Compute sigmoid activation
    for (int i = 0; i < n_features; i++) {
        fx[b * n_features + i] = 1.0 / (1.0 + exp(-x[b * n_features + i]));
    }
}

// Sigmoid gradient on batch - y(b, n_features) = fx * (1 - fx) (b, n_features)
void sigmoid_gradient_batch(const double *fx, double *dfx, const int size, const int n_features) {
    // Compute sigmoid gradient
    const int batch_size = size / n_features;
    for (int b = 0; b < batch_size; b++) {
        sigmoid_gradient(b, fx, dfx, n_features);
    }
}

// Sigmoid gradient - y = fx * (1 - fx)
void sigmoid_gradient(const int b, const double *fx, double *dfx, const int n_features) {
    // Compute sigmoid gradient
    for (int i = 0; i < n_features; i++) {
        dfx[b * n_features + i] *= (fx[b * n_features + i] * (1 - fx[b * n_features + i]));
    }
}
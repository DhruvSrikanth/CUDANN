#include "softmax.h"
#include "../utils/tensor.h"
#include <cmath>
#include <cstring>

// Initialize the SigmSoftmaxoid Layer
Softmax::Softmax(const int n_classes, const std::string name) {
    // Set layer name
    this->name = name;

    // Set the number of classes
    this->n_classes = n_classes;
    
    // Initialize input, output, input gradient and output gradient
    this->x = NULL;
    this->fx = NULL;
    this->dfx = NULL;
}

// Destructor
Softmax::~Softmax() {
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
void Softmax::show() {
    std::printf("%s: [%d]\n", this->name.c_str(), this->n_classes);
}

// Forward call
Tensor* Softmax::forward(const Tensor *input) {
    const int size = input->n_batches * this->n_classes;

    // Allocate memory for input, output and input gradient
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
    create_tensor(this->fx, input->n_batches, this->n_classes);

    // Compute softmax transformation
    softmax_activation_batch(this->x->data, this->fx->data, size, this->n_classes);
    
    // Return output
    return this->fx;
}

// Backward call
Tensor* Softmax::backward(const Tensor *upstream_grad) {
    const int size = upstream_grad->n_batches * this->n_classes;

    // Copy upstream gradient to dfx
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
    this->dfx = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->dfx, (const Tensor*) this->fx);

    // Compute softmax gradient
    softmax_gradient_batch(upstream_grad->data, this->fx->data, this->dfx->data, size, this->n_classes);
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
        softmax_gradient(b, upstream_grad, fx, dfx, n_classes);
    }
}

// Softmax gradient - y = fx * (upstream_grad - fx * upstream_grad)
void softmax_gradient(const int b, const double* upstream_grad, const double *fx, double *dfx, const int n_classes) {
    // Compute softmax gradient
    for (int i = 0; i < n_classes; i++) {
        dfx[b * n_classes + i] *= (upstream_grad[b * n_classes + i] - (upstream_grad[b * n_classes + i] * fx[b * n_classes + i]));
    }
}
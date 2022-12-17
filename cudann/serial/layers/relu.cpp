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
    this->x = NULL;
    this->fx = NULL;
    this->dfx = NULL;
}
    

// Destructor
ReLU::~ReLU() {
    // Free memory
    free_tensor(this->x);
    free_tensor(this->fx);
    free_tensor(this->dfx); 
}




// Print layer
void ReLU::show() {
    std::printf("%s: [%d]\n", this->name.c_str(), this->n_features);
}

Tensor* ReLU::forward(const Tensor *input) {
    const int size = input->n_batches * this->n_features;

    // Allocate memory for input, output and input gradient
    if (this->x != NULL) {
        free_tensor(this->x);

    }
    this->x = (Tensor*) malloc(sizeof(Tensor));
    // Copy input to x
    copy_tensor(this->x, (Tensor*) input);
    
    if (this->fx != NULL) {
        free_tensor(this->fx);
    }
    this->fx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->fx, input->n_batches, this->n_features);


    // Compute relu activation
    relu_activation_batch(this->x->data, this->fx->data, size, this->n_features);

    // Return output
    return this->fx;
}

// Backward call
Tensor* ReLU::backward(const Tensor *upstream_grad) {
    const int size = upstream_grad->n_batches * this->n_features;

    // Copy upstream gradient to dfx
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }

    this->dfx = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->dfx, (Tensor*) upstream_grad);

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

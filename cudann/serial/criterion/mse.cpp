#include "mse.h"
#include <math.h>

// Constructor
MSE::MSE(const std::string name) {
    this->name = name;

    this->fx = NULL;
    this->grad = NULL;        
}

// Destructor
MSE::~MSE() {
    if (this->fx != NULL) {
        free(this->fx);
    }
    if (this->grad != NULL) {
        free(this->grad);
    }
}

// Compute mse
Tensor *MSE::forward(const Tensor *input, const Tensor *target) {
    const int size = input->n_batches * input->n_features;
    
    // Allocate memory for input and gradient
    if (this->fx != NULL) {
        free(this->fx);
    }
    this->fx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->fx, input->n_batches, 1);

    if (this->grad != NULL) {
        free(this->grad);
    }
    this->grad = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->grad, input->n_batches, input->n_features);

    // Compute mse
    mse_batch(input->data, target->data, this->fx->data, input->n_features, input->n_batches);

    // Compute mse gradient
    mse_grad_batch(input->n_batches, input->data, target->data, this->grad->data, input->n_features);

    // Return output
    return this->fx;
}

// Compute mse gradient
Tensor *MSE::backward() {
    return this->grad;
}

// Print the loss name
void MSE::show() {
    std::printf("%s\n", this->name.c_str());
}


// Compute mse for a batch
void mse_batch(const double *input, const double *target, double *fx, const int n_features, const int n_batches) {
    for (int b = 0; b < n_batches; b++) {
        // Initialize output
        fx[b*n_features] = 0.0;

        mse(b, input, target, fx, n_features);

        // Compute mean
        fx[b*n_features] /= n_features;
    }
}

// Compute mse
void mse(const int b, const double *input, const double *target, double *fx, const int n_features) {
    for (int i = 0; i < n_features; i++) {
        fx[b*n_features] += (0.5 * pow(input[b*n_features + i] - target[b*n_features + i], 2));
    }
    
}

// Compute mse gradient for a batch
void mse_grad_batch(const int n_batches, const double *input, const double *target, double* grad, const int n_features) {
    for (int b = 0; b < n_batches; b++) {
        mse_grad(b, input, target, grad, n_features);
    }
}

// Compute mse gradient
void mse_grad(const int b, const double *input, const double *target, double* grad, const int n_features) {
    for (int i = 0; i < n_features; i++) {
        grad[b*n_features + i] = (input[b*n_features + i] - target[b*n_features + i]) / n_features;
    }
}
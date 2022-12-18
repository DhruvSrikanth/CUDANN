#include "ce.h"
#include <math.h>

// Constructor
CrossEntropy::CrossEntropy(const std::string name) {
    this->name = name;

    this->input = NULL;
    this->target = NULL;

    this->fx = NULL;
    this->grad = NULL;        
}

// Destructor
CrossEntropy::~CrossEntropy() {
    if (this->input != NULL) {
        free(this->input);
    }
    if (this->target != NULL) {
        free(this->target);
    }
    if (this->fx != NULL) {
        free(this->fx);
    }
    if (this->grad != NULL) {
        free(this->grad);
    }
}

// Compute cross entropy
Tensor *CrossEntropy::forward(const Tensor *input, const Tensor *target) {
    const int size = input->n_batches * input->n_features;

    // Allocate memory for input, target, output and gradient and copy input and target
    if (this->input != NULL) {
        free(this->input);
    }
    this->input = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->input, input);

    if (this->target != NULL) {
        free(this->target);
    }
    this->target = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->target, target);

    // Clip target
    const double min = 1e-8;
    const double *max = nullptr;
    this->target->clip(&min, max);

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

    // Compute cross entropy
    ce_batch(input->data, target->data, this->fx->data, input->n_features, input->n_batches);

    // Compute cross entropy gradient
    ce_grad_batch(input->data, target->data, this->grad->data, input->n_features, input->n_batches);

    return this->fx;
}

// Compute cross entropy gradient
Tensor *CrossEntropy::backward() {
    return this->grad;
}

// Compute cross entropy for a batch
void ce_batch(const double *input, const double *target, double *fx, const int n_features, const int n_batches) {
    for (int b = 0; b < n_batches; b++) {
        // Reset cross entropy for a single sample
        fx[b] = 0.0;

        // Compute cross entropy for a single sample
        ce(b, input, target, fx, n_features);

        // Compute mean across features
        fx[b] /= n_features;
    }
}

// Compute cross entropy for a single sample
void ce(const int b, const double *input, const double *target, double *fx, const int n_features) {
    // (np.where(self.y == 1, -np.log(self.y_hat), 0)).sum(axis=1)
    for (int i = 0; i < n_features; i++) {
        if (target[b * n_features + i] == 1.0) {
            fx[b] += -log(input[b * n_features + i]);
        }
    }
}

// Compute cross entropy gradient for a batch
void ce_grad_batch(const double *input, const double *target, double *grad, const int n_features, const int n_batches) {
    for (int b = 0; b < n_batches; b++) {
        ce_grad(b, input, target, grad, n_features);
    }
}

// Compute cross entropy gradient for a single sample
void ce_grad(const int b, const double *input, const double *target, double *grad, const int n_features) {
    // np.where(self.y == 1, -1 / self.y_hat, 0)
    for (int i = 0; i < n_features; i++) {
        if (target[b * n_features + i] == 1.0) {
            grad[b * n_features + i] = -1.0 / input[b * n_features + i];
        } else {
            grad[b * n_features + i] = 0.0;
        }
    }
}


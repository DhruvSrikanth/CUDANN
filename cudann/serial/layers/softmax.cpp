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
    this->downstream_grad = NULL;
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
    if (this->downstream_grad != NULL) {
        free_tensor(this->downstream_grad);
    }
}

// Print layer name
void Softmax::show() {
    std::printf("%s: [%d]\n", this->name.c_str(), this->n_classes);
}

// Get layer type
std::string Softmax::get_type() {
    return this->type;
}

// Forward call
Tensor* Softmax::forward(const Tensor *input) {
    const int size = input->batch_size * this->n_classes;

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
    create_tensor(this->fx, input->batch_size, this->n_classes);

    // Compute softmax transformation
    softmax_activation_batch(this->x->data, this->fx->data, size, this->n_classes);
    
    // Return output
    return this->fx;
}

// Backward call
Tensor* Softmax::backward(const Tensor *upstream_grad) {
    const int size = upstream_grad->batch_size * this->n_classes;

    // Copy upstream gradient to dfx
    if (this->downstream_grad != NULL) {
        free_tensor(this->downstream_grad);
    }
    this->downstream_grad = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->downstream_grad, upstream_grad->batch_size, this->n_classes);

    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
    this->dfx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->dfx, upstream_grad->batch_size, this->n_classes * this->n_classes);

    // Compute softmax gradient
    softmax_gradient_batch(upstream_grad->data, this->fx->data, this->downstream_grad->data, this->dfx->data, size, this->n_classes);

    // Return output
    return this->downstream_grad;
}
    
// Softmax activation on batch - y(b, n_classes) = e^(x(b, n_classes)) / sum(e^(x(b, n_classes))) (b, n_classes)
void softmax_activation_batch(const double *x, double *fx, const int size, const int n_classes) {
    // Perform softmax activation on each batch
    const int batch_size = size / n_classes;
    for (int b = 0; b < batch_size; b++) {
        softmax_activation(b, x, fx, n_classes);
    }
}

// Softmax activation - y = e^(x) / sum(e^(x))
void softmax_activation(const int b, const double *x, double *fx, const int n_classes) {
    // Compute max x for each batch (for numerical stability)
    double max_x = x[b * n_classes];
    for (int i = 0; i < n_classes; i++) {
        if (x[b * n_classes + i] > max_x) {
            max_x = x[b * n_classes + i];
        }
    }

    // Compute softmax activation
    double exp_sum = 0;
    for (int i = 0; i < n_classes; i++) {
        fx[b * n_classes + i] = exp(x[b * n_classes + i] - max_x);
        exp_sum += fx[b * n_classes + i];
    }
    for (int i = 0; i < n_classes; i++) {
        fx[b * n_classes + i] /= exp_sum;
    }
}

// Softmax gradient on batch - y(b, n_classes) = fx(b, n_classes) * (upstream_grad(b, n_classes) - fx(b, n_classes) * upstream_grad(b, n_classes)) (b, n_classes)
void softmax_gradient_batch(const double* upstream_grad, const double *fx, double* downstream_grad, double *dfx, const int size, const int n_classes) {
    // Compute softmax gradient
    const int batch_size = size / n_classes;
    for (int b = 0; b < batch_size; b++) {
        softmax_gradient(b, upstream_grad, fx, downstream_grad, dfx, n_classes);
    }
}

// Softmax gradient - y = fx * upstream_grad * (1 - fx) when i = j, y = -fx * upstream_grad * fx when i != j
void softmax_gradient(const int b, const double* upstream_grad, const double *fx, double *downstream_grad, double *dfx, const int n_classes) {
    // Compute softmax gradient
    for (int i = 0; i < n_classes; i++) {
        for (int j = 0; j < n_classes; j++) {
            if (i == j) {
                dfx[(b * n_classes * n_classes) + (i * n_classes) + j] = fx[b * n_classes + i] * (1 - fx[b * n_classes + j]);
            } else {
                dfx[(b * n_classes * n_classes) + (i * n_classes) + j] = -fx[b * n_classes + i] * fx[b * n_classes + j];
            }
        }
    }

    // Compute downstream gradient
    for (int i = 0; i < n_classes; i++) {
        downstream_grad[b * n_classes + i] = 0;
        for (int j = 0; j < n_classes; j++) {
            downstream_grad[b * n_classes + i] += dfx[(b * n_classes * n_classes) + (i * n_classes) + j] * upstream_grad[b * n_classes + j];
        }
    }
}
#include "layer.h"

void Layer::show() {
    std::printf("Error: Method must be implemented in child class\n");
}

Tensor* Layer::forward(const Tensor *input) {
    std::printf("Error: Method must be implemented in child class\n");
    return (Tensor*) input;
}

Tensor* Layer::backward(const Tensor *upstream_grad) {
    std::printf("Error: Method must be implemented in child class\n");
    return (Tensor*) upstream_grad;
}

void Layer::update_params(const double learning_rate) {
    std::printf("Error: Method must be implemented in child class\n");
    return;
}

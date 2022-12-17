#ifndef LAYER_H
#define LAYER_H

#include "../utils/tensor.h"
class Layer {
    public:
        std::string type;
        void show();
        Tensor* forward(const Tensor *input);
        Tensor* backward(const Tensor *upstream_grad);
        void update_params(const double learning_rate);
};

#endif
#ifndef LAYER_H
#define LAYER_H

#include "../utils/tensor.h"
class Layer {
    public:
        void show();
        Tensor* forward(const Tensor *input);
        Tensor* backward(const Tensor *upstream_grad);
};

#endif
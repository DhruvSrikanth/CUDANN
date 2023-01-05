#ifndef LAYER_H
#define LAYER_H

#include "../utils/tensor.h"
#include <string>

class Layer {
    public:
        std::string type;
        virtual std::string get_type();

        virtual void show();
        
        virtual Tensor* forward(const Tensor *input);
        virtual Tensor* backward(const Tensor *upstream_grad);
        virtual void update_params(const double learning_rate);
};

#endif
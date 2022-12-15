#include <iostream>
#include <string.h> 

#ifndef Layer_H
#define Layer_H

class Layer {
    public:
        void show();
        double* forward(const double *input);
        double* backward(const double *upstream_grad);
        void update_params(const double learning_rate);
};

#endif
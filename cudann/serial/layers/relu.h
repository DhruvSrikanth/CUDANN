#ifndef RELU_H
#define RELU_H

#include <iostream>
#include "layer.h"
#include "../utils/tensor.h"

void relu_activation_batch(const double *x, double *fx, const int batch_size, const int n_features);
void relu_activation(const int b, const double *x, double *fx, const int n_features);
void relu_gradient_batch(const double *x, double *dfx, const int batch_size, const int n_features);
void relu_gradient(const int b, const double *x, double *dfx, const int n_features);


class ReLU: public Layer {
    public:
        // Initialize the ReLU Layer
        int n_features;
        Tensor *x;
        Tensor *fx;
        Tensor *dfx;

        // Layer name
        std::string name;
    
        // Initialize the ReLU Layer
        ReLU(const int n_features, const std::string name="ReLU");
        
        // Destructor
        ~ReLU();

        // Print layer information
        void show();

        // Forward call
        Tensor* forward(const Tensor *input);

        // Backward call
        Tensor* backward(const Tensor *upstream_grad);
};

#endif
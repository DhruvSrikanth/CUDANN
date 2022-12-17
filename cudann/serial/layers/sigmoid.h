#ifndef SIGMOID_H
#define SIGMOID_H

#include <iostream>
#include "layer.h"
#include "../utils/tensor.h"

void sigmoid_activation_batch(const double *x, double *fx, const int batch_size, const int n_features);
void sigmoid_activation(const int b, const double *x, double *fx, const int n_features);
void sigmoid_gradient_batch(const double *fx, double *dfx, const int batch_size, const int n_features);
void sigmoid_gradient(const int b, const double *fx, double *dfx, const int n_features);

class Sigmoid: public Layer {
    public:
        // Layer type
        std::string type = "Sigmoid";
        
        // Layer name
        std::string name;

        // Input, output, input gradient, output gradient
        Tensor *x;
        Tensor *fx;
        Tensor *dfx;

        // Number of features
        int n_features;

        // Initialize the Sigmoid Layer
        Sigmoid(const int n_features, const std::string name="Sigmoid");

        // Destructor
        ~Sigmoid();

        // Print layer name
        void show();

        // Forward call
        Tensor* forward(const Tensor *input);

        // Backward call
        Tensor* backward(const Tensor *upstream_grad);
};

#endif
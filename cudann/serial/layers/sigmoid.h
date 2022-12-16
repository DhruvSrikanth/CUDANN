#ifndef SIGMOID_H
#define SIGMOID_H

#include <iostream>
#include "layer.h"

void sigmoid_activation_batch(const double *x, double *fx, const int batch_size, const int n_features);
void sigmoid_activation(const int b, const double *x, double *fx, const int n_features);
void sigmoid_gradient_batch(const double *fx, double *dfx, const int batch_size, const int n_features);
void sigmoid_gradient(const int b, const double *fx, double *dfx, const int n_features);

class Sigmoid: public Layer {
    public:
        // Layer name
        std::string name;

        // Input, output, input gradient, output gradient
        double *x;
        double *fx;
        double *dfx;

        // Number of features
        int n_features;

        // Initialize the Sigmoid Layer
        Sigmoid(const int n_features, const std::string name="Sigmoid");

        // Destructor
        ~Sigmoid();

        // Print layer name
        void show();

        // Forward call
        double* forward(const double *input);

        // Backward call
        double* backward(const double *upstream_grad);
};

#endif
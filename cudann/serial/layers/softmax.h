#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <iostream>
#include "layer.h"

void softmax_activation_batch(const double *x, double *fx, const int batch_size, const int n_features);
void softmax_activation(const int b, const double *x, double *fx, const int n_features);
void softmax_gradient_batch(const double* upstream_grad, const double *fx, double *dfx, const int size, const int n_classes);
void softmax_gradient(const int b, const double* upstream_grad, const double *fx, double *dfx, const int n_classes);

class Softmax: public Layer {
    public:
        // Layer name
        std::string name;

        // Input, output, input gradient, output gradient
        double *x;
        double *fx;
        double *dfx;

        // Number of features
        int n_classes;

        // Initialize the Softmax Layer
        Softmax(const int n_classes, const std::string name="Softmax");

        // Destructor
        ~Softmax();

        // Print layer name
        void show();

        // Forward call
        double* forward(const double *input);

        // Backward call
        double* backward(const double *upstream_grad);
};

#endif
#ifndef RELU_H
#define RELU_H

#include <iostream>
#include "layer.h"

void relu_activation_batch(const double *x, double *fx, const int batch_size, const int n_features);
void relu_activation(const int b, const double *x, double *fx, const int n_features);
void relu_gradient_batch(const double *x, double *dfx, const int batch_size, const int n_features);
void relu_gradient(const int b, const double *x, double *dfx, const int n_features);


class ReLU: public Layer {
    public:
        // Initialize the ReLU Layer
        int n_features;
        double *x;
        double *fx;
        double *dfx;

        // Layer name
        std::string name;
    
    // Initialize the ReLU Layer
    ReLU(const int n_features, const std::string name="ReLU");
    
    // Destructor
    ~ReLU();

    // Print layer information
    void show();

    // Forward call
    double* forward(const double *input);

    // Backward call
    double* backward(const double *upstream_grad);
};

#endif
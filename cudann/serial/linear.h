#include <stdio.h>
#include <iostream>
#include <string.h> 

#ifndef Linear_H
#define Linear_H

class Linear {
    public:
        // Perceptron Layer size
        int in_features;
        int out_features;

        // Bias flag
        bool bias_flag;

        // Weight initialization technique
        std::string initialization;

        // Layer name
        std::string name;

        // Weights and bias
        double *weight;
        double *bias;

        // Input, output, input gradient, weight gradient and bias gradient
        double *x;
        double *fx;
        double *dfx;
        double *dW;
        double *db;
        double *mean_dW;
        double *mean_db;
    
        // Initialize the Perceptron Layer
        Linear(const int in_features, const int out_features, const bool bias, const std::string initialization, const std::string name);

        // Destructor
        ~Linear();

        // Print layer name
        void show();

        // Forward call
        double* forward(const double *input);

        // Backward call
        double* backward(const double *upstream_grad);

        // Update weights
        void update_params(const double learning_rate);
};

#endif
#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"
#include "../utils/tensor.h"
#include <string>

void linear_transformation_batch(const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag, const int n_batches);
void linear_transformation(const int b, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag);
void linear_transformation_gradient_batch(const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool bias_flag, const int n_batches);
void linear_transformation_gradient(const int b, const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool bias_flag, const int n_batches);

class Linear: public Layer {
    public:
        // Layer type
        std::string type = "Linear";

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
        Tensor *x;
        Tensor *fx;
        Tensor *dfx;
        
        double *dW;
        double *db;
        double *mean_dW;
        double *mean_db;
    
        // Initialize the Perceptron Layer
        Linear(const int in_features, const int out_features, const bool bias, const std::string initialization, const std::string name="Linear");

        // Destructor
        ~Linear();

        // Print layer name
        void show();

        // Forward call
        Tensor* forward(const Tensor *input);

        // Backward call
        Tensor* backward(const Tensor *upstream_grad);

        // Update weights
        void update_params(const double learning_rate);
};

#endif
#ifndef MSE_H
#define MSE_H

#include "../utils/tensor.h"
#include <string>

void mse_batch(const double *input, const double *target, double *fx, const int n_features, const int n_batches);
void mse(const int b, const double *input, const double *target, double *fx, const int n_features);
void mse_grad_batch(const int n_batches, const double *input, const double *target, double* grad, const int n_features);
void mse_grad(const int b, const double *input, const double *target, double* grad, const int n_features);

class MSE {
    public:
        // Attributes
        Tensor* fx;
        Tensor* grad;
        
        std::string name;

        
        // Constructor
        MSE(const std::string name="mse");

        // Destructor
        ~MSE();

        // Print the loss name
        void show();

        // Forward pass
        Tensor *forward(const Tensor *input, const Tensor *target);

        // Backward pass
        Tensor *backward();
};

#endif
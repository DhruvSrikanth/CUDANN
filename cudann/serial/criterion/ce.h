#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H


#include "../utils/tensor.h"
#include <string>


void ce_batch(const double *input, const double *target, double *fx, const int n_features, const int n_batches);
void ce(const int b, const double *input, const double *target, double *fx, const int n_features);
void ce_grad_batch(const double *input, const double *target, double *grad, const int n_features, const int n_batches);
void ce_grad(const int b, const double *input, const double *target, double *grad, const int n_features);


class CrossEntropy {
    public:
        // Attributes
        Tensor *input;
        Tensor *target;
        
        Tensor* fx;
        Tensor* grad;
        
        std::string name;

        
        // Constructor
        CrossEntropy(const std::string name="crossentropy");

        // Destructor
        ~CrossEntropy();

        // Forward pass
        Tensor *forward(const Tensor *input, const Tensor *target);

        // Backward pass
        Tensor *backward();
};

#endif
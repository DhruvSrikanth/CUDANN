#include <iostream>
#include <string.h> 

#ifndef Softmax_H
#define Softmax_H

class Softmax {
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
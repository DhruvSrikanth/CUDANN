#include <iostream>
#include <string.h> 
#include <cmath>

#ifndef Sigmoid_H
#define Sigmoid_H

class Sigmoid {
    public:
        // Layer name
        std::string name;

        // Input, output, input gradient, output gradient
        double *x;
        double *fx;
        double *dfx;
        bool first_call = true;

        // Initialize the Sigmoid Layer
        Sigmoid(std::string name="Sigmoid");

        // Print layer name
        void show();

        // Forward call
        double* forward(const double *input);

        // Backward call
        double* backward(const double *upstream_grad);
};

#endif
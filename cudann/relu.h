#include <stdio.h>
#include <iostream>
#include <string.h>

#ifndef ReLU_H
#define ReLU_H

class ReLU {
    public:
        // Initialize the ReLU Layer
        double *x;
        double *fx;
        double *dfx;

        bool first_call = false;

        // Layer name
        std::string name;

    ReLU(const std::string name="ReLU");

    // Print layer information
    void show();

    // Forward call
    double* forward(const double *input);

    // Backward call
    double* backward(const double *upstream_grad);
};

#endif
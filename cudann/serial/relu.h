#include <iostream>
#include <string.h>

#ifndef ReLU_H
#define ReLU_H

class ReLU {
    public:
        // Initialize the ReLU Layer
        int n_features;
        double *x;
        double *fx;
        double *dfx;

        // Layer name
        std::string name;

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
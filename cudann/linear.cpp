#include <stdio.h>
#include <iostream>
#include <string.h> 
#include <math.h>
#include "initialize.h"
#include "linear.h"


// Initialize the Perceptron Layer
Linear::Linear(const int in_features, const int out_features, const bool bias, const std::string initialization, const std::string name="Linear") {
    // Set layer size
    this->in_features = in_features;
    this->out_features = out_features;

    // Initialize weights
    this->initialization = initialization;
    // Output neurons are columns and input neurons are rows, therefore weight matrix is of shape (in_features, out_features)
    this->weight = new double[this->in_features * this->out_features];
    initialize_weights(&this->weight, this->in_features, this->out_features, this->initialization);

        // Set bias flag and initialize bias
    this->bias_flag = bias;
    if (this->bias_flag) {
        this->bias = new double[this->out_features];
        initialize_bias(&this->bias, this->out_features);
    }

    // Set layer name
    this->name = name;

    // Initialize the weight and bias gradients
    initialize_zeros(&this->dW, this->in_features * this->out_features);
    if (this->bias_flag) {
        initialize_zeros(&this->db, this->out_features);
    }
}

// Print layer information
void Linear::show() {
    std::printf("%s: [%d -> %d]\n", this->name, this->in_features, this->out_features);
}

// Forward call
double* Linear::forward(const double *input) {
    // Copy input to x
    const int in_size = sizeof(input) / sizeof(input[0]);
    this->x = new double[in_size];
    memcpy(this->x, input, in_size * sizeof(double));

    // Allocate memory for output and output gradient
    const int out_size = this->out_features * in_size / this->in_features;
    this->fx = new double[out_size];
    this->dfx = new double[out_size];

    // Compute linear transformation on the batch
    linear_transformation_batch(this->out_features, this->in_features, this->weight, this->x, this->bias, this->fx, this->bias_flag, in_size);

    // Return output
    return this->fx;
}

// Backward call
double* Linear::backward(const double *upstream_grad) {
    // Compute linear transformation gradient
    linear_transformation_grad(upstream_grad, &this->out_features, &this->in_features, this->weight, this->x, this->bias, this->dfx, this->dW, this->db, &this->bias_flag);

    // Return output gradient
    return this->dfx;
}

// Update parameters
void Linear::update_params(const double lr) {
    // Update weights
    for (int i = 0; i < this->in_features; i++) {
        for (int j = 0; j < this->out_features; j++) {
            this->weight[i * this->out_features + j] -= lr * this->dW[i * this->out_features + j];
        }
    }

    // Update bias
    if (this->bias_flag) {
        for (int i = 0; i < this->out_features; i++) {
            this->bias[i] -= lr * this->db[i];
        }
    }
}

// Linear transformation on a batch - fx(b, out) = W(in x out) x x(b, out) + bias (b, out)
void linear_transformation_batch(const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag, const int in_size) {
    const int batch_size = in_size / in_features;
    
    // Compute linear transformation on the batch
    for (int b = 0; b < batch_size; b++) {
        // Compute linear transformation for each example
        linear_transformation(b, out_features, in_features, weight, x, bias, fx, bias_flag, in_size);
    }
}

// Linear transformation - fx(out) = W(in x out) x x(in) + bias (out)
void linear_transformation(const int b, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag, const int in_size) {
    // Compute linear transformation
    for (int i = 0; i < out_features; i++) {

        // Reset fx for each output neuron
        fx[b * out_features + i] = 0.0;

        // Get the dot product of weight and input
        for (int j = 0; j < in_features; j++) {
            fx[b * out_features + i] += weight[(i * in_features)+ j] * x[b * in_features + j];
        }

        // Add bias
        if (bias_flag) {
            fx[b * out_features + i] += bias[b * out_features + i];
        }
    }
}

// Linear transformation gradient - dfx(1, in) = dfx(1, out) x W(out x in)
void linear_transformation_grad(const double *upstream_grad, const int *out_features, const int *in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool *bias_flag) {
    // Compute linear transformation gradient
    // dfx(1, in) = dfx(1, out) x W(out x in)
    for (int i = 0; i < *in_features; i++) {
        // Reset dfx for each input neuron
        dfx[i] = 0.0;

        // Get the dot product of upstream gradient and weight
        for (int j = 0; j < *out_features; j++) {
            dfx[i] += upstream_grad[j] * weight[(i * *out_features)+ j];
        }
    }

    // Compute weight gradient
    // dW(in x out) = x(1, in) x dfx(1, out)
    // self.dW = (np.matmul(self.x[:, :, None], upstream_grad[:, None, :])).mean(axis=0)
    for (int i = 0; i < *in_features; i++) {
        for (int j = 0; j < *out_features; j++) {
            dW[(i * *out_features) + j] = x[i] * upstream_grad[j];
        }
    }


    // Compute bias gradient
    if (*bias_flag) {
        // db(1, out) = dfx(1, out)
        memcpy(db, upstream_grad, *out_features * sizeof(double));
    }
}
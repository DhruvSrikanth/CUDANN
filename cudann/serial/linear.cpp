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
    this->weight = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    initialize_weights(&this->weight, this->in_features, this->out_features, this->initialization);

    // Set bias flag and initialize bias
    this->bias_flag = bias;
    if (this->bias_flag) {
        this->bias = (double*) malloc(this->out_features * sizeof(double));
        initialize_bias(&this->bias, this->out_features);
    }

    // Set layer name
    this->name = name;

    // Initialize the weight, bias and mean gradients
    this->dW = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    this->mean_dW = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    if (this->bias_flag) {
        this->db = (double*) malloc(this->out_features * sizeof(double));
        this->mean_db = (double*) malloc(this->out_features * sizeof(double));
    }

    // Initialize the input, output and input gradient
    this->x = (double*) malloc(this->in_features * sizeof(double));
    this->fx = (double*) malloc(this->out_features * sizeof(double));
    this->dfx = (double*) malloc(this->in_features * sizeof(double));    
}

// Destructor
Linear::~Linear() {
    free(this->weight);
    free(this->dW);
    free(this->mean_dW);
    if (this->bias_flag) {
        free(this->bias);
        free(this->db);
        free(this->mean_db);
    }
    free(this->x);
    free(this->fx);
    free(this->dfx);
}

// Print layer information
void Linear::show() {
    std::printf("%s: [%d -> %d]\n", this->name, this->in_features, this->out_features);
}

// Forward call
double* Linear::forward(const double *input) {
    const int n_batches = (sizeof(input) / sizeof(input[0])) / this->in_features;
    // Reallocate batch size memory to x and Copy input to x
    this->x = (double*) realloc(this->x, this->in_features * n_batches * sizeof(double));
    memcpy(this->x, input, this->in_features * n_batches * sizeof(double));

    // Allocate memory for output
    this->fx = (double*) realloc(this->fx, this->out_features * n_batches * sizeof(double));

    // Compute linear transformation on the batch
    linear_transformation_batch(this->out_features, this->in_features, this->weight, this->x, this->bias, this->fx, this->bias_flag);

    // Return output
    return this->fx;
}

// Backward call
double* Linear::backward(const double *upstream_grad) {
    // Allocate memory for input gradient and the weight and bias gradients
    const int n_batches = (sizeof(upstream_grad) / sizeof(upstream_grad[0])) / this->out_features;
    this->dfx = (double*) realloc(this->dfx, this->in_features * n_batches * sizeof(double));
    this->dW = (double*) realloc(this->dW, this->in_features * this->out_features * n_batches * sizeof(double));
    initialize_zeros(&this->dW, this->in_features * this->out_features * n_batches);
    if (this->bias_flag) {
        this->db = (double*) realloc(this->db, this->out_features * n_batches * sizeof(double));
        initialize_zeros(&this->db, this->out_features * n_batches);
    }
    
    // Compute linear transformation gradient
    linear_transformation_gradient_batch(upstream_grad, this->out_features, this->in_features, this->weight, this->x, this->bias, this->dfx, this->dW, this->db, this->bias_flag, n_batches);

    // Return output gradient
    return this->dfx;
}

// Update parameters of the layer and reset gradients
void Linear::update_params(const double lr) {
    const int n_batches = (sizeof(this->x) / sizeof(this->x[0])) / this->in_features;

    // Sum gradients over the batch and store in mean_dW and mean_db
    for (int b = 0; b < n_batches; b++) {
        for (int i = 0; i < this->in_features; i++) {
            for (int j = 0; j < this->out_features; j++) {
                this->mean_dW[i * this->out_features + j] += this->dW[(b * this->in_features * this->out_features) + (i * this->out_features) + j] / n_batches;
            }
        }

        if (this->bias_flag) {
            for (int i = 0; i < this->out_features; i++) {
                this->mean_db[i] += this->db[(b * this->out_features) + i] / n_batches;
            }
        }
    }


    // Update weights
    for (int i = 0; i < this->in_features; i++) {
        for (int j = 0; j < this->out_features; j++) {
            this->weight[i * this->out_features + j] -= lr * this->mean_dW[i * this->out_features + j];
            // Reset mean_dW
            this->mean_dW[i * this->out_features + j] = 0;
        }
    }

    // Update bias
    if (this->bias_flag) {
        for (int i = 0; i < this->out_features; i++) {
            this->bias[i] -= lr * this->mean_db[i];
            // Reset mean_db
            this->mean_db[i] = 0.0;
        }
    }
}

// Linear transformation on a batch - fx(b, out) = W(in x out) x x(b, out) + bias (b, out)
void linear_transformation_batch(const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag) {
    const int n_batches = (sizeof(x) / sizeof(x[0])) / in_features;
    
    // Compute linear transformation on the batch
    for (int b = 0; b < n_batches; b++) {
        // Compute linear transformation for each example
        linear_transformation(b, out_features, in_features, weight, x, bias, fx, bias_flag);
    }
}

// Linear transformation - fx(out) = W(in x out) x x(in) + bias (out)
void linear_transformation(const int b, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag) {
    // Compute linear transformation
    for (int i = 0; i < out_features; i++) {
        // Reset fx for each output neuron
        fx[b * out_features + i] = 0.0;

        // Compute linear transformation
        for (int j = 0; j < in_features; j++) {
            fx[b * out_features + i] += weight[j * out_features + i] * x[b * in_features + j];
        }
        // Add bias
        if (bias_flag) {
            fx[b * out_features + i] += bias[i];
        }
    }
}

// Linear transformation gradient batch - dfx(b, in) = dfx(b`, out) x W(out x in)
void linear_transformation_gradient_batch(const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool bias_flag, const int n_batches) {    
    // Compute linear transformation gradient for the batch
    for (int b = 0; b < n_batches; b++) {
        // Compute linear transformation gradient for each example
        linear_transformation_gradient(b, upstream_grad, out_features, in_features, weight, x, bias, dfx, dW, db, bias_flag, n_batches);
    }
}

// Linear transformation gradient - dfx(in) = dfx(out) x W(out x in)
void linear_transformation_gradient(const int b, const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool bias_flag, const int n_batches) {
    // dfx(1, in) = dfx(1, out) x W(out x in)
    for (int i = 0; i < in_features; i++) {
        // Reset dfx for each input neuron
        dfx[b * in_features + i] = 0.0;

        // Get the dot product of upstream gradient and weight
        for (int j = 0; j < out_features; j++) {
            dfx[b * in_features + i] += upstream_grad[b * out_features + j] * weight[(i * out_features) + j];
        }
    }

    // Compute weight gradient
    // dW(in x out) = x(1, in) x dfx(1, out)
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j++) {
            dW[(b * in_features * out_features) + (i * out_features) + j] += x[b * in_features + i] * upstream_grad[b * out_features + j];
        }
    }


    // Compute bias gradient
    if (bias_flag) {
        // db(1, out) = dfx(1, out)
        for (int i = 0; i < out_features; i++) {
            db[b * out_features + i] += upstream_grad[b * out_features + i];
        }
    }
}
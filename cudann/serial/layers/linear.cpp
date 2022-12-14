#include "linear.h"
#include "../utils/tensor.h"
#include "../utils/initialize.h"
#include <cstring>
#include <random>

// Initialize the Perceptron Layer
Linear::Linear(const int in_features, const int out_features, const bool bias, const std::string initialization, const std::string name) {
    // Set layer size
    this->in_features = in_features;
    this->out_features = out_features;

    // Initialize weights
    this->initialization = initialization;
    
    // Output neurons are columns and input neurons are rows, therefore weight matrix is of shape (in_features, out_features)
    this->weight = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    initialize_weights(this->weight, this->in_features, this->out_features, this->initialization);
    // Set bias flag and initialize bias
    this->bias_flag = bias;
    if (this->bias_flag) {
        this->bias = (double*) malloc(this->out_features * sizeof(double));
        initialize_bias(this->bias, this->out_features);
    }

    // Set layer name
    this->name = name;

    // Initialize the weight, bias and mean gradients
    this->dW = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    this->mean_dW = (double*) malloc(this->in_features * this->out_features * sizeof(double));
    initialize_zeros(this->mean_dW, this->in_features * this->out_features);
    if (this->bias_flag) {
        this->db = (double*) malloc(this->out_features * sizeof(double));
        this->mean_db = (double*) malloc(this->out_features * sizeof(double));
        initialize_zeros(this->mean_db, this->out_features);
    }

    // Initialize the input, output and input gradient
    this->x = NULL;
    this->fx = NULL;
    this->dfx = NULL;
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
    if (this->x != NULL) {
        free_tensor(this->x);
    }
    if (this->fx != NULL) {
        free_tensor(this->fx);
    }
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
}

// Print layer information
void Linear::show() {
    std::printf("%s: [%d -> %d]\n", this->name.c_str(), this->in_features, this->out_features);
}

// Get layer type
std::string Linear::get_type() {
    return this->type;
}

// Forward call
Tensor* Linear::forward(const Tensor *input) {
    // Size of the input
    const int size = input->batch_size * input->n_features;

    // Allocate batch size memory to x and copy input to x
    if (this->x != NULL) {
        free_tensor(this->x);
    }
    this->x = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(this->x, input);

    // Allocate memory for output
    if (this->fx != NULL) {
        free_tensor(this->fx);
    }
    this->fx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->fx, input->batch_size, this->out_features);

    // Compute linear transformation on the batch
    linear_transformation_batch(this->out_features, this->in_features, this->weight, this->x->data, this->bias, this->fx->data, this->bias_flag, input->batch_size);

    // Return output
    return this->fx;
}

// Backward call
Tensor* Linear::backward(const Tensor *upstream_grad) {
    // Allocate memory for input gradient and the weight and bias gradients
    const int batch_size = upstream_grad->batch_size;
    if (this->dfx != NULL) {
        free_tensor(this->dfx);
    }
    this->dfx = (Tensor*) malloc(sizeof(Tensor));
    create_tensor(this->dfx, batch_size, this->in_features);
    
    this->dW = (double*) realloc(this->dW, this->in_features * this->out_features * batch_size * sizeof(double));
    initialize_zeros(this->dW, this->in_features * this->out_features * batch_size);
    if (this->bias_flag) {
        this->db = (double*) realloc(this->db, this->out_features * batch_size * sizeof(double));
        initialize_zeros(this->db, this->out_features * batch_size);
    }
    
    // Compute linear transformation gradient
    linear_transformation_gradient_batch(upstream_grad->data, this->out_features, this->in_features, this->weight, this->x->data, this->bias, this->dfx->data, this->dW, this->mean_dW, this->db, this->mean_db, this->bias_flag, batch_size);

    // Return output gradient
    return this->dfx;
}

// Update parameters of the layer and reset gradients
void Linear::update_params(const double lr) {
    // Update weights
    for (int i = 0; i < this->in_features; i++) {
        for (int j = 0; j < this->out_features; j++) {
            this->weight[i * this->out_features + j] -= lr * this->mean_dW[i * this->out_features + j];
            // Reset mean_dW
            this->mean_dW[i * this->out_features + j] = 0.0;
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
void linear_transformation_batch(const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* fx, const bool bias_flag, const int batch_size) {
    // Compute linear transformation on the batch
    for (int b = 0; b < batch_size; b++) {
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
            fx[b * out_features + i] += x[b * in_features + j] * weight[i * in_features + j];
        }

        // Add bias
        if (bias_flag) {
            fx[b * out_features + i] += bias[i];
        }
    }

    
}

// Linear transformation gradient batch - dfx(b, in) = dfx(b, out) x W(out x in)
void linear_transformation_gradient_batch(const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double *mean_dW, double* db, double *mean_db, const bool bias_flag, const int batch_size) {    
    // Compute linear transformation gradient for the batch
    for (int b = 0; b < batch_size; b++) {
        // Compute linear transformation gradient for each example
        linear_transformation_gradient(b, upstream_grad, out_features, in_features, weight, x, bias, dfx, dW, db, bias_flag);
    }

    // Sum gradients over the batch and store in mean_dW and mean_db
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < in_features; i++) {
            for (int j = 0; j < out_features; j++) {
                mean_dW[i * out_features + j] += dW[(b * in_features * out_features) + (i * out_features) + j] / batch_size;
            }
        }

        if (bias_flag) {
            for (int i = 0; i < out_features; i++) {
                mean_db[i] += db[(b * out_features) + i] / batch_size;
            }
        }
    }
}

// Linear transformation gradient - dfx(in) = dfx(out) x W(out x in)
void linear_transformation_gradient(const int b, const double *upstream_grad, const int out_features, const int in_features, const double* weight, const double* x, const double* bias, double* dfx, double* dW, double* db, const bool bias_flag) {
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
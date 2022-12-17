#include "tensor.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>

// Tensor Constructor
Tensor::Tensor(const int n_batches, const int n_features, const double *data) {
    this->n_batches = n_batches;
    this->n_features = n_features;
    this->data = (double *) malloc(n_batches * n_features * sizeof(double));
    memcpy(this->data, data, n_batches * n_features * sizeof(double));
}

// Tensor Destructor
Tensor::~Tensor() {
    free(this->data);
}

// Print tensor
void Tensor::show() {
    printf("Tensor: [%d batches, %d features, %d size, %lu bytes]\n", this->n_batches, this->n_features, this->n_batches * this->n_features, this->n_batches * this->n_features * sizeof(double));
}

// Free memory
void free_tensor(Tensor *tensor) {
    free(tensor->data);
    free(tensor);
}

// Copy tensor
void copy_tensor(Tensor *dst, Tensor *src) {
    dst->n_batches = src->n_batches;
    dst->n_features = src->n_features;
    dst->data = (double *) malloc(src->n_batches * src->n_features * sizeof(double));
    memcpy(dst->data, src->data, src->n_batches * src->n_features * sizeof(double));
}

// Initialize tensor
void initialize_tensor(Tensor *tensor, const int n_batches, const int n_features, const double *data) {
    tensor->n_batches = n_batches;
    tensor->n_features = n_features;
    tensor->data = (double *) malloc(n_batches * n_features * sizeof(double));
    memcpy(tensor->data, data, n_batches * n_features * sizeof(double));
}

// Create tensor
void create_tensor(Tensor *tensor, const int n_batches, const int n_features) {
    tensor->n_batches = n_batches;
    tensor->n_features = n_features;
    tensor->data = (double *) malloc(n_batches * n_features * sizeof(double));
}

// Clip tensor
void Tensor::clip(const double min, const double max) {
    for (int b = 0; b < this->n_batches; b++) {
        for (int f = 0; f < this->n_features; f++) {
            if (min != NULL) {
                if (this->data[b * this->n_features + f] < min) {
                    this->data[b * this->n_features + f] = min;
                }
            }
            if (max != NULL) {
                if (this->data[b * this->n_features + f] > max) {
                    this->data[b * this->n_features + f] = max;
                }
            }
        }
    }
}

// Print tensor
void Tensor::print() {
    for (int b = 0; b < this->n_batches; b++) {
        for (int f = 0; f < this->n_features; f++) {
            printf("%f ", this->data[b * this->n_features + f]);
        }
        printf("\n");
    }
}
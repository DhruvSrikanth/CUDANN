#include "tensor.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>

// Tensor Constructor
Tensor::Tensor(const int batch_size, const int n_features, const double *data) {
    this->batch_size = batch_size;
    this->n_features = n_features;
    this->data = (double *) malloc(batch_size * n_features * sizeof(double));
    memcpy(this->data, data, batch_size * n_features * sizeof(double));
}

// Tensor Destructor
Tensor::~Tensor() {
    free(this->data);
}

// Print tensor
void Tensor::show() {
    printf("Tensor: [%d samples, %d features, %d size, %lu bytes]\n", this->batch_size, this->n_features, this->batch_size * this->n_features, this->batch_size * this->n_features * sizeof(double));
}

// Free memory
void free_tensor(Tensor *tensor) {
    free(tensor->data);
    free(tensor);
}

// Copy tensor
void copy_tensor(Tensor *dst, const Tensor *src) {
    dst->batch_size = src->batch_size;
    dst->n_features = src->n_features;
    dst->data = (double *) malloc(src->batch_size * src->n_features * sizeof(double));
    memcpy(dst->data, src->data, src->batch_size * src->n_features * sizeof(double));
}

// Initialize tensor
void initialize_tensor(Tensor *tensor, const int batch_size, const int n_features, const double *data) {
    tensor->batch_size = batch_size;
    tensor->n_features = n_features;
    tensor->data = (double *) malloc(batch_size * n_features * sizeof(double));
    memcpy(tensor->data, data, batch_size * n_features * sizeof(double));
}

// Create tensor
void create_tensor(Tensor *tensor, const int batch_size, const int n_features) {
    tensor->batch_size = batch_size;
    tensor->n_features = n_features;
    tensor->data = (double *) malloc(batch_size * n_features * sizeof(double));
}

// Clip tensor
void Tensor::clip(const double* min, const double* max) {
    for (int b = 0; b < this->batch_size; b++) {
        for (int f = 0; f < this->n_features; f++) {
            if (min != nullptr) {
                if (this->data[b * this->n_features + f] < *min) {
                    this->data[b * this->n_features + f] = *min;
                }
            }
            if (max != nullptr) {
                if (this->data[b * this->n_features + f] > *max) {
                    this->data[b * this->n_features + f] = *max;
                }
            }
        }
    }
}

// Print tensor
void Tensor::print() {
    printf("\n[");
    for (int b = 0; b < this->batch_size; b++) {
        printf("\n[");
        for (int f = 0; f < this->n_features; f++) {
            printf("%f", this->data[b * this->n_features + f]);
            if (f < this->n_features - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (b < this->batch_size - 1) {
            printf(", ");
        }
    }
    printf("\n]\n");
}

// Sum tensor data
double Tensor::sum() {
    double sum_ = 0.0;
    for (int b = 0; b < this->batch_size; b++) {
        for (int f = 0; f < this->n_features; f++) {
            sum_ += this->data[b * this->n_features + f];
        }
    }
    return sum_;
}
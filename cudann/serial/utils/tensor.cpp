#include "tensor.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
    print_array(this->data, this->batch_size, this->n_features);
}

// Print double array
void print_array(const double *data, const int batch_size, const int n_features) {
    printf("\n[");
    for (int b = 0; b < batch_size; b++) {
        printf("\n[");
        for (int f = 0; f < n_features; f++) {
            printf("%f", data[b * n_features + f]);
            if (f < n_features - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (b < batch_size - 1) {
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

// Check array for nan values
bool check_array_for_nan(const double *array, const int size) {
    for (int i = 0; i < size; i++) {
        if (std::isnan(array[i])) {
            return true;
        }
    }
    return false;
}

// Check tensor for nan values
bool Tensor::has_nan() {
    return check_array_for_nan(this->data, this->batch_size * this->n_features);
}

// Get the argmax of the tensor
void Tensor::argmax(int *b_argmax) {
    for (int b = 0; b < this->batch_size; b++) {
        double max = this->data[b * this->n_features];
        int argmax = 0;
        for (int f = 1; f < this->n_features; f++) {
            if (this->data[b * this->n_features + f] > max) {
                max = this->data[b * this->n_features + f];
                argmax = f;
            }
        }
        b_argmax[b] = argmax;
    }
}
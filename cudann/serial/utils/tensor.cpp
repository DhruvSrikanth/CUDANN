#include "tensor.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>

Tensor::Tensor(const int n_batches, const int n_features, const double *data) {
    this->n_batches = n_batches;
    this->n_features = n_features;
    this->data = (double *) malloc(n_batches * n_features * sizeof(double));
    memcpy(this->data, data, n_batches * n_features * sizeof(double));
}

Tensor::~Tensor() {
    free(this->data);
}

void Tensor::show() {
    printf("Tensor: [%d batches, %d features, %d size, %lu bytes]\n", this->n_batches, this->n_features, this->n_batches * this->n_features, this->n_batches * this->n_features * sizeof(double));
}


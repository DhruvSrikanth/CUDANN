#include <stdio.h>
#include <iostream>
#include <string.h>
#include <random>
#include "initialize.h"

void initialize_bias(double **bias, const int out_features) {
    for (int i = 0; i < out_features; i++) {
        (*bias)[i] = 0.0;
    }
}

void initialize_weights(double **weights, const int in_features, const int out_features, const std::string type) {
    if (type == "random") {
        random_initialization(weights, in_features, out_features);
    }
    else if (type == "uniform") {
        uniform_initialization(weights, in_features, out_features);
    } else {
        throw std::invalid_argument("Invalid weight initialization - " + type + ". Consult documentation for valid types.");
    }
}

void random_initialization(double **weights, const int in_features, const int out_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j++) {
            (*weights)[i * out_features + j] = 0.0;
            (*weights)[i * out_features + j] = d(gen) * sqrt(2 / in_features);
        }
    }
}

void uniform_initialization(double **weights, const int in_features, const int out_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(-1, 1);
    for (int i = 0; i < in_features; i++) {
        for (int j = 0; j < out_features; j++) {
            (*weights)[i * out_features + j] = 0.0;
            (*weights)[i * out_features + j] = d(gen) * sqrt(2 / in_features);
        }
    }
}

void initialize_zeros(double **x, const int size) {
    for (int i = 0; i < size; i++) {
        (*x)[i] = 0.0;
    }
}
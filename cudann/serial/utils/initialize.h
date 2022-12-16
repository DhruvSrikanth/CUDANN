#include <iostream>
void initialize_bias(double **bias, const int out_features);
void initialize_weights(double **weights, const int in_features, const int out_features, const std::string type);
void random_initialization(double **weights, const int in_features, const int out_features);
void uniform_initialization(double **weights, const int in_features, const int out_features);
void initialize_zeros(double **x, const int size);
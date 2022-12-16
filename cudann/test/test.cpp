#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 

#include "../serial/utils/tensor.h"

int main(int argc, char *argv[]) {
    // Add layers
    int n_classes = 10;
    int n_features = 784;
    int n_batches = 2;
    double learning_rate = 0.01;

    // ReLU layer
    // ReLU relu(n_features, "relu");
    // relu.show();

    // // Random input tensor
    double *input = (double *) malloc(n_batches * n_features * sizeof(double));
    srand(time(0));
    for(int i = 0; i < n_batches * n_features; i++){
        input[i] = (rand() % 100) - 50;
    }
    Tensor input_tensor(n_batches, n_features, input);
    input_tensor.show();

    // // // Forward pass
    // tensor *output = relu.forward(&input_tensor);

    // Print output as a matrix
    // for(int b = 0; b < n_batches; b++){
    //     for(int i = 0; i < n_features; i++){
    //         std::cout << output->data[b * n_features + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }

    // Free memory
    // free(input);



    return 0;
}


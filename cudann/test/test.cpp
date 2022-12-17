#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 

#include "../serial/utils/tensor.h"
#include "../serial/layers/softmax.h"

int main(int argc, char *argv[]) {
    // Add layers
    int n_classes = 10;
    int n_features = 28*28;
    int n_batches = 2;
    double learning_rate = 0.01;

    // Softmax layer
    Softmax softmax(n_classes, "Softmax");
    softmax.show();

    // Random input tensor
    double *input = (double *) malloc(n_batches * n_features * sizeof(double));
    for (int i = 0; i < n_batches * n_features; i++) {
        input[i] = i - (n_batches * n_features / 2);
    }
    Tensor input_tensor(n_batches, n_features, input);

    // Forward pass
    Tensor *output = softmax.forward(&input_tensor);

    // Print output as a matrix
    // for(int b = 0; b < n_batches; b++){
    //     for(int i = 0; i < n_features; i++){
    //         std::cout << output->data[b * n_features + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }
    

    // Backward pass
    double *upstream_grad = (double *) malloc(n_batches * n_features * sizeof(double));
    for (int i = 0; i < n_batches * n_features; i++) {
        if (i - (n_batches * n_features / 2) > 0) {
            upstream_grad[i] = 1;
        } else {
            upstream_grad[i] = 0;
        }
    }
    Tensor upstream_grad_tensor(n_batches, n_features, upstream_grad);
    Tensor *input_grad = softmax.backward(&upstream_grad_tensor);

    // Print output as a matrix
    // for(int b = 0; b < n_batches; b++){
    //     for(int i = 0; i < n_features; i++){
    //         std::cout << input_grad->data[b * n_features + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }





    return 0;
}


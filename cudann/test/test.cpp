#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 

#include "../serial/utils/tensor.h"
#include "../serial/utils/initialize.h"
#include "../serial/layers/linear.h"

int main(int argc, char *argv[]) {
    // Add layers
    int n_classes = 10;
    int n_features = 28*28;
    int n_batches = 2;
    double learning_rate = 0.01;

    // Softmax layer
    Linear linear(n_features, n_classes, true, "random", "Linear");
    linear.show();

    // Random input tensor
    double *input = (double *) malloc(n_batches * n_features * sizeof(double));
    initialize_weights(input, n_batches, n_features, "random");
    Tensor input_tensor(n_batches, n_features, input);

    // Forward pass
    Tensor *output = linear.forward(&input_tensor);

    // Print output as a matrix
    // for(int b = 0; b < n_batches; b++){
    //     for(int i = 0; i < n_classes; i++){
    //         std::cout << output->data[b * n_classes + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }
    

    // Backward pass
    double *upstream_grad = (double *) malloc(n_batches * n_classes * sizeof(double));
    initialize_weights(upstream_grad, n_batches, n_classes, "random");
    Tensor upstream_grad_tensor(n_batches, n_classes, upstream_grad);
    
    Tensor *input_grad = linear.backward(&upstream_grad_tensor);

    // Print output as a matrix
    // for(int b = 0; b < n_batches; b++){
    //     for(int i = 0; i < n_features; i++){
    //         std::cout << input_grad->data[b * n_features + i] << ",";
    //     }
    //     std::cout << std::endl;
    // }





    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 

#include "../serial/utils/tensor.h"
#include "../serial/utils/initialize.h"
#include "../serial/layers/linear.h"
#include "../serial/layers/softmax.h"
#include "../serial/layers/relu.h"
#include "../serial/model/nn.h"
#include "../serial/criterion/mse.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 28*28;
    const int n_batches = 2;
    const double learning_rate = 0.01;

    // // Add layers
    // Linear linear1(n_features, 128, true, "random", "linear1");
    // ReLU relu1(128, "relu1");
    // Linear linear2(128, n_classes, true, "random", "linear2");
    // Softmax softmax(n_classes, "softmax");

    // // Create model and add layers
    // NN model;
    // model.add_layer(&linear1);
    // model.add_layer(&relu1);
    // model.add_layer(&linear2);
    // model.add_layer(&softmax);

    // // Print model summary
    // model.summary();

    // // Create random input tensor
    // double *data = (double*) malloc(n_batches*n_features*sizeof(double));
    // for (int i = 0; i < n_batches*n_features; i++) {
    //     data[i] = (double) rand() / (double) RAND_MAX;
    // }
    // Tensor input(n_batches, n_features, data);
    
    // // Forward pass
    // Tensor *output = model.forward(&input);

    // // Create random upstream gradient tensor
    // double *grad_data = (double*) malloc(n_batches*n_classes*sizeof(double));
    // for (int i = 0; i < n_batches*n_classes; i++) {
    //     grad_data[i] = (double) rand() / (double) RAND_MAX;
    // }
    // Tensor upstream_grad(n_batches, n_classes, grad_data);

    // // Backward pass
    // Tensor *downstream_grad = model.backward(&upstream_grad);

    // // Update weights
    // model.update_weights(learning_rate);

    // Test loss function
    // Create random input tensor of 1s and 0s
    double *data = (double*) malloc(n_batches*n_classes*sizeof(double));
    initialize_salt_and_pepper(data, n_batches*n_classes);
    Tensor input(n_batches, n_classes, data);


    // Create random target tensor of 1s and 0s
    double *target_data = (double*) malloc(n_batches*n_classes*sizeof(double));
    initialize_salt_and_pepper(target_data, n_batches*n_classes);
    Tensor target(n_batches, n_classes, target_data);

    // Create loss function
    MSE mse("mse");

    // Compute loss
    Tensor *loss = mse.forward(&input, &target);
    // loss->print();
    // Average loss
    double avg_loss = 0.0;
    for (int i = 0; i < n_batches; i++) {
        avg_loss += loss->data[i];
    }
    avg_loss /= n_batches;
    std::cout << "Average loss: " << avg_loss << std::endl;

    // Compute gradient
    Tensor *grad = mse.backward();
    // grad->print();


    return 0;
}


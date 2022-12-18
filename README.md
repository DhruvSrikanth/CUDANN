# CUDANN
A distributed implementation of a deep learning framework in CUDA.

# Serial Implementation

This can be tested using the following command - 

```shell
make test
```

A working example of the framework can be seen below - 

```c++
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
#include "../serial/criterion/ce.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 28*28;
    const int n_batches = 2;
    const double learning_rate = 0.01;

    // Add layers
    Linear linear1(n_features, 128, true, "random", "linear1");
    ReLU relu1(128, "relu1");
    Linear linear2(128, n_classes, true, "random", "linear2");
    Softmax softmax(n_classes, "softmax");

    // Create model and add layers
    NN model;
    model.add_layer(&linear1);
    model.add_layer(&relu1);
    model.add_layer(&linear2);
    model.add_layer(&softmax);

    // Print model summary
    model.summary();

    // Get loss function
    CrossEntropy criterion("cross_entropy");
    printf("Loss function: ");
    criterion.show();

    // Create random input tensor
    double *data = (double*) malloc(n_batches*n_features*sizeof(double));
    for (int i = 0; i < n_batches*n_features; i++) {
        data[i] = (double) rand() / (double) RAND_MAX;
    }
    Tensor input(n_batches, n_features, data);

    // Create the random target tensor
    double *target_data = (double*) malloc(n_batches*n_classes*sizeof(double));
    initialize_salt_and_pepper(target_data, n_batches*n_classes);
    Tensor target(n_batches, n_classes, target_data);
    
    // Forward pass
    Tensor *output = model.forward(&input);

    // Compute loss
    Tensor *loss = criterion.forward(output, &target);
    
    // Compute the loss gradient
    Tensor *downstream_grad = criterion.backward();

    // Backward pass
    model.backward(downstream_grad);

    // Update weights
    model.update_weights(learning_rate);

    // Compute average loss
    double avg_loss = loss->sum() / (double) n_batches;
    printf("Average loss: %f\n", avg_loss);

    return 0;
}
```

This is the output of the above code - 

```bash
=========================
Model - NN
=========================
linear1: [784 -> 128]
relu1: [128]
linear2: [128 -> 10]
softmax: [10]
==========================
Loss function: cross_entropy
Average loss: 1.329639
```
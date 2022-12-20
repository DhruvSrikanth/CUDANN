# CUDANN
A distributed implementation of a deep learning framework in CUDA.

# Serial Implementation

This can be tested using the following command - 

```shell
make test_random
```

The above command runs the following example of using the framework - 

```c++
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 
#include <unordered_map>

#include "../serial/cudann.h"


struct MiniBatch {
    Tensor *input;
    Tensor *target;
};

struct Dataloader {
    MiniBatch *minibatches;
    int n_batches;
};


void train(const int n_classes, const int n_features, const int n_batches, const double learning_rate, const int epochs, Dataloader *dataloader, NN *model, CrossEntropy *criterion) {
    // Print model summary
    model->summary();

    printf("Loss function: ");
    criterion->show();

    printf("Training model for %d epochs with learning rate %f.\n", epochs, learning_rate);
    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Compute average loss over a batch
        double avg_loss = 0.0;
        for (int mb = 0; mb < dataloader->n_batches; mb++) {
            // Get minibatch and copy it to the appropriate device
            Tensor *input = (Tensor*) malloc(sizeof(Tensor));
            Tensor *target = (Tensor*) malloc(sizeof(Tensor));
            copy_tensor(input, dataloader->minibatches[mb].input);
            copy_tensor(target, dataloader->minibatches[mb].target);

            // Forward pass
            Tensor *output = model->forward(input);

            // Compute loss
            Tensor *loss = criterion->forward(output, target);
            
            // Compute the loss gradient
            Tensor *downstream_grad = criterion->backward();

            // Backward pass
            model->backward(downstream_grad);

            // Update weights
            model->update_weights(learning_rate);

            // Compute average loss
            avg_loss += loss->sum() / loss->batch_size;
        }
        avg_loss /= dataloader->n_batches;
        
        printf("Epoch %d: Average loss: %f\n", epoch + 1, avg_loss);
    }
}

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 28*28;
    const int batch_size = 2;
    const double learning_rate = 0.01;
    const int n_batches = 10;
    const int epochs = 10;

    // Add layers
    Linear linear1(n_features, 10, true, "random", "linear1");
    ReLU relu1(128, "relu1");
    Linear linear2(128, n_classes, true, "random", "linear2");
    Softmax softmax(n_classes, "softmax");

    // Create model and add layers
    NN model;
    model.add_layer(&linear1);
    model.add_layer(&relu1);
    model.add_layer(&linear2);
    model.add_layer(&softmax);

    // Get loss function
    CrossEntropy criterion("cross_entropy");

    // Create dataloader
    Dataloader dataloader;
    dataloader.n_batches = n_batches;
    dataloader.minibatches = (MiniBatch*) malloc(n_batches*sizeof(MiniBatch));
    for (int i = 0; i < n_batches; i++) {
        // Create random input tensor
        double *data = (double*) malloc(batch_size*n_features*sizeof(double));
        initialize_random(data, batch_size*n_features);
        Tensor input(batch_size, n_features, data);

        // Create the random target tensor
        double *target_data = (double*) malloc(batch_size*n_classes*sizeof(double));
        initialize_salt_and_pepper(target_data, batch_size*n_classes);
        Tensor target(batch_size, n_classes, target_data);

        // Create minibatch
        MiniBatch minibatch;
        minibatch.input = (Tensor*) malloc(sizeof(Tensor));
        minibatch.target = (Tensor*) malloc(sizeof(Tensor));
        copy_tensor(minibatch.input, &input);
        copy_tensor(minibatch.target, &target);

        // Add minibatch to dataloader
        dataloader.minibatches[i] = minibatch;
    }

    // Train the model
    train(n_classes, n_features, n_batches, learning_rate, epochs, &dataloader, &model, &criterion);

    
    return 0;
}


```

This is the output of the above code - 

```bash
=========================
Model - NN
=========================
linear1: [784 -> 10]
relu1: [128]
linear2: [128 -> 10]
softmax: [10]
==========================
Loss function: cross_entropy
Training model for 10 epochs with learning rate 0.010000.
Epoch 1: Average loss: 1.164237
Epoch 2: Average loss: 1.171686
Epoch 3: Average loss: 1.171686
Epoch 4: Average loss: 1.169180
Epoch 5: Average loss: 1.166300
Epoch 6: Average loss: 1.184660
Epoch 7: Average loss: 1.180385
Epoch 8: Average loss: 1.162772
Epoch 9: Average loss: 1.156481
Epoch 10: Average loss: 1.156554
```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 
#include <cmath>

#include "../serial/cudann.h"


struct MiniBatch {
    Tensor *input;
    Tensor *target;
};

struct Dataloader {
    MiniBatch *minibatches;
    int n_batches;
};

// Check array for nan values
bool check_array_for_nan(const double *array, const int size) {
    for (int i = 0; i < size; i++) {
        if (std::isnan(array[i])) {
            return true;
        }
    }
    return false;
}

void train(const int n_classes, const int n_features, const double learning_rate, const int epochs, Dataloader *dataloader, NN *model, CrossEntropy *criterion) {
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
    const int batch_size = 64;
    const double learning_rate = 0.01;
    const int n_batches = 1000;
    const int epochs = 5;

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
    train(n_classes, n_features, learning_rate, epochs, &dataloader, &model, &criterion);

    
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 
#include <unordered_map>


#include "../serial/cudann.h"

struct Dataloader {
    std::unordered_map<std::string, const Tensor*> *minibatches;
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
            // Get minibatch
            std::unordered_map<std::string, const Tensor*> minibatch = dataloader->minibatches[mb];
            Tensor input = *minibatch["input"];
            Tensor target = *minibatch["target"];

            // Forward pass
            Tensor *output = model->forward(&input);
            std::cout << "Reached here\n";

            // Compute loss
            Tensor *loss = criterion->forward(output, &target);
            
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

    // Data loader as an array of minibatches
    std::unordered_map<std::string, const Tensor*> *minibatches = (std::unordered_map<std::string, const Tensor*>*) malloc(n_batches*sizeof(std::unordered_map<std::string, const Tensor*>));
    for (int i = 0; i < n_batches; i++) {
        // Create random input tensor
        double *data = (double*) malloc(batch_size*n_features*sizeof(double));
        for (int i = 0; i < batch_size*n_features; i++) {
            data[i] = (double) rand() / (double) RAND_MAX;
        }
        Tensor input(batch_size, n_features, data);

        // Create the random target tensor
        double *target_data = (double*) malloc(batch_size*n_classes*sizeof(double));
        initialize_salt_and_pepper(target_data, batch_size*n_classes);
        Tensor target(batch_size, n_classes, target_data);

        // Create minibatch
        std::unordered_map<std::string, const Tensor*> minibatch;
        minibatch["input"] = &input;
        minibatch["target"] = &target;

        minibatches[i] = minibatch;
    }
    Dataloader dataloader = {minibatches, n_batches};

    // Train the model
    train(n_classes, n_features, n_batches, learning_rate, epochs, &dataloader, &model, &criterion);

    
    return 0;
}


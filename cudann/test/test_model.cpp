#include "../serial/cudann.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 28*28;
    const int batch_size = 2;
    const double learning_rate = 0.01;
    const int n_batches = 1;
    const int epochs = 5;
    std::string test_type = "forward";

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

    // Create loss
    CrossEntropy criterion("cross_entropy");

    // Create input tensor
    double *data = (double*) malloc(batch_size*n_features*sizeof(double));
    initialize_random(data, batch_size*n_features);
    Tensor *input = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(input, batch_size, n_features, data);
    if (test_type == "forward" || test_type == "backward") {
        input->print();
    }
    

    // Forward pass
    Tensor* output = model.forward(input);
    if (test_type == "forward") {
        output->print();
    }

    // Create target tensor
    double *target_data = (double*) malloc(batch_size*n_classes*sizeof(double));
    initialize_salt_and_pepper(target_data, batch_size*n_classes);
    Tensor *target = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(target, batch_size, n_classes, target_data);
    if (test_type == "forward" || test_type == "backward") {
        target->print();
    }

    // Compute loss
    Tensor *loss = criterion.forward(output, target);
    if (test_type == "forward") {
        loss->print();
    }
    
    // Compute the loss gradient
    Tensor *downstream_grad = criterion.backward();

    // Backward pass
    model.backward(downstream_grad);

    // Update weights
    model.update_weights(learning_rate);

    return 0;
}
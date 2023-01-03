#include "../serial/cudann.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 10;
    const int batch_size = 2;
    std::string test_type = "backward";

    // Create a model
    NN model;
    Sigmoid layer(n_features, "layer1");
    double *data = (double*) malloc(n_features * batch_size * sizeof(double));
    initialize_random(data, n_features * batch_size);
    Tensor *input = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(input, batch_size, n_features, data);
    if (test_type == "forward" || test_type == "backward") {
        input->print();
    }

    // Forward pass
    Tensor* output = layer.forward(input);
    if (test_type == "forward") {
        output->print();
    }

    // Backward pass
    double *grad = (double*) malloc(n_features * batch_size * sizeof(double));
    initialize_random(grad, n_features * batch_size);
    Tensor *grad_output = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(grad_output, batch_size, n_features, grad);
    if (test_type == "backward") {
        grad_output->print();
    }

    Tensor *grad_input = layer.backward(grad_output);
    if (test_type == "backward") {
        grad_input->print();
    }

    return 0;
}
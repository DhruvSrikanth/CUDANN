#include "../serial/cudann.h"
#include <cstring>

int main(int argc, char *argv[]) {
    const int out_features = 3;
    const int in_features = 2;
    const int batch_size = 2;
    std::string test_type = "backward";

    // Create a model
    NN model;
    Linear layer(in_features, out_features, true, "random", "layer1");
    double *data = (double*) malloc(in_features * batch_size * sizeof(double));
    initialize_random(data, in_features * batch_size);
    Tensor *input = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(input, batch_size, in_features, data);
    if (test_type == "forward" || test_type == "backward") {
        input->print();
    }

    // Forward pass
    Tensor* output = layer.forward(input);
    if (test_type == "forward") {
        output->print();
    }

    // Backward pass
    double *grad = (double*) malloc(out_features * batch_size * sizeof(double));
    initialize_random(grad, out_features * batch_size);
    Tensor *grad_output = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(grad_output, batch_size, out_features, grad);
    if (test_type == "backward") {
        grad_output->print();
    }

    Tensor *grad_input = layer.backward(grad_output);
    if (test_type == "backward") {
        grad_input->print();
    }

    return 0;
}
#include "../serial/cudann.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int n_features = 784;

    // Create a model
    NN model;
    ReLU relu(n_features, "relu1");
    double *data = (double*) malloc(n_features*sizeof(double));
    initialize_random(data, n_features);
    Tensor *input = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(input, 1, n_features, data);
    input->print();

    // Forward pass
    Tensor* output = relu.forward(input);
    output->print();

    return 0;
}
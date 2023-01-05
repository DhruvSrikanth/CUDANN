#include "../serial/cudann.h"

int main(int argc, char *argv[]) {
    const int n_classes = 10;
    const int batch_size = 2;
    std::string test_type = "backward";

    // Create output tensor
    double *data = (double*) malloc(batch_size*n_classes*sizeof(double));
    initialize_random(data, batch_size*n_classes);
    Tensor *output = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(output, batch_size, n_classes, data);

    // Pass through softmax
    Softmax softmax(n_classes, "softmax");
    Tensor *probs = softmax.forward(output);
    if (test_type == "forward" || test_type == "backward") {
        printf("Probabilities - ");
        probs->print();
    }

    // Create target tensor
    double *target_data = (double*) malloc(batch_size*n_classes*sizeof(double));
    initialize_salt_and_pepper(target_data, batch_size*n_classes);
    Tensor *target = (Tensor*) malloc(sizeof(Tensor));
    initialize_tensor(target, batch_size, n_classes, target_data);
    if (test_type == "forward" || test_type == "backward") {
        printf("Target - ");
        target->print();
    }

    // Create loss
    CrossEntropy criterion("cross_entropy");

    // Compute loss
    Tensor *loss = criterion.forward(probs, target);
    if (test_type == "forward") {
        printf("Loss - ");
        loss->print();
    }
    
    // Compute the loss gradient
    Tensor *downstream_grad = criterion.backward();
    if (test_type == "backward") {
        printf("Downstream gradient - ");
        downstream_grad->print();
    }

    return 0;
}
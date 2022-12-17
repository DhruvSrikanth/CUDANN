#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <time.h> 

#include "../serial/utils/tensor.h"
#include "../serial/layers/linear.h"
#include "../serial/layers/softmax.h"
#include "../serial/layers/relu.h"
#include "../serial/model/nn.h"

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

    return 0;
}


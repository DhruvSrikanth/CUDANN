#include "nn.h"
#include "../utils/tensor.h"
#include "../layers/layer.h"

// Initialize the neural network
NN::NN(const std::string name) {
    this->name = name;
    this->n_layers = 0;
    this->layers = (Layer**) calloc(0, sizeof(Layer*));
}

// Destructor
NN::~NN() {
    free(this->layers);
}

// Print summary of the network
void NN::summary() {
    std::printf("=========================\n");
    std::printf("Model - %s\n", this->name.c_str());
    std::printf("=========================\n");
    for (int i = 0; i < this->n_layers; i++) {
        this->layers[i]->show();
    }
    std::printf("==========================\n");
}

// Add layer
void NN::add_layer(const Layer *layer) {
    this->n_layers++;

    this->layers = (Layer**) realloc(this->layers, this->n_layers * sizeof(Layer*));
    this->layers[this->n_layers - 1] = (Layer*) layer;
}

// Forward call
Tensor* NN::forward(const Tensor *input) {
    Tensor *output = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(output, (Tensor*) input);

    for (int i = 0; i < this->n_layers; i++) {
        output = this->layers[i]->forward(output);
    }
    return output;
}

// Backward call
Tensor* NN::backward(const Tensor *upstream_grad) {
    Tensor *downstream_grad = (Tensor*) malloc(sizeof(Tensor));
    copy_tensor(downstream_grad, (Tensor*) upstream_grad);

    for (int i = this->n_layers - 1; i >= 0; i--) {
        downstream_grad = this->layers[i]->backward(downstream_grad);
    }
    return downstream_grad;
}

// Update weights
void NN::update_weights(const double learning_rate) {
    for (int i = 0; i < this->n_layers; i++) {
        if (this->layers[i]->type == "Linear") {
            this->layers[i]->update_params(learning_rate);
        }
    }
}


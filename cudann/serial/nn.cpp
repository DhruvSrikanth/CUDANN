#include <iostream>
#include <string.h>
#include "layer.h"
#include "nn.h"

// Initialize the neural network
NN::NN(const std::string name) {
    this->name = name;
    this->n_layers = 0;
    Layer *layer = (Layer*) calloc(0, sizeof(Layer));
    this->layers = &layer;
}

// Destructor
NN::~NN() {
    free(this->layers);
}

// Print summary of the network
void NN::summary() {
    std::printf("Model: %s\n", this->name);
    std::printf("Layers: \n");
    for (int i = 0; i < this->n_layers; i++) {
        std::printf("Layer %d: ", i);
        this->layers[i]->show();
    }
    std::printf("\n");
}

// Add layer
void NN::add_layer(const Layer *layer, const std::string layer_type) {
    this->n_layers++;
    this->layers = (Layer**) realloc(this->layers, this->n_layers * sizeof(Layer*));
    this->layers[this->n_layers - 1] = (Layer*) layer;
    this ->layer_types[this->n_layers - 1] = layer_type;
}

// Forward call
double* NN::forward(const double *input) {
    double *output;
    for (int i = 0; i < this->n_layers; i++) {
        output = this->layers[i]->forward(input);
        input = output;
    }
    return output;
}

// Backward call
double* NN::backward(const double *upstream_grad) {
    double *downstream_grad;
    for (int i = this->n_layers - 1; i >= 0; i--) {
        downstream_grad = this->layers[i]->backward(upstream_grad);
        upstream_grad = downstream_grad;
    }
    return downstream_grad;
}

// Update weights
void NN::update_weights(const double learning_rate) {
    for (int i = 0; i < this->n_layers; i++) {
        if (this->layer_types[i] == "Linear") {
            this->layers[i]->update_params(learning_rate);
        }
    }
}


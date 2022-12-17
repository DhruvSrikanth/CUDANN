#ifndef NN_H
#define NN_H

#include <iostream>
#include "../layers/layer.h"


class NN {
    public:
        // Model name
        std::string name;

        // Number of layers
        int n_layers;

        // Pointer to layers
        Layer **layers;

        // Layer types
        std::string *layer_types;

        // Initialize the Softmax Layer
        NN(const std::string name="NN");

        // Destructor
        ~NN();

        // Print the network
        void summary();

        // Add layer
        void add_layer(const Layer *layer, const std::string layer_type);

        // Forward call
        Tensor* forward(const Tensor *input);

        // Backward call
        Tensor* backward(const Tensor *upstream_grad);

        // Update weights
        void update_weights(const double learning_rate);
};


#endif
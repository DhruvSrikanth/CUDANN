#include <stdio.h>
#include <iostream>
#include <string.h> 
#include <math.h>
#include "initialize.h"

class Linear {
    public:
        // Perceptron Layer size
        int in_features;
        int out_features;

        // Bias flag
        bool bias_flag;

        // Weight initialization technique
        std::string initialization;

        // Layer name
        std::string name;

        // Weights and bias
        double *weight;
        double *bias;

        // Input, output, input gradient, weight gradient and bias gradient
        double *x;
        double *fx;
        double *dfx;
        double *dW;
        double *db;

        // Initialize the Perceptron Layer
        Linear(int in_features, int out_features, bool bias, std::string initialization, std::string name="Linear") {
            // Set layer size
            this->in_features = in_features;
            this->out_features = out_features;

            // Initialize weights
            this->initialization = initialization;
            // Output neurons are columns and input neurons are rows, therefore weight matrix is of shape (in_features, out_features)
            this->weight = new double[this->in_features * this->out_features];
            initialize_weights(&this->weight, this->in_features, this->out_features, this->initialization);

             // Set bias flag and initialize bias
            this->bias_flag = bias;
            if (this->bias_flag) {
                this->bias = new double[this->out_features];
                initialize_bias(&this->bias, this->out_features);
            }

            // Set layer name
            this->name = name;

            // Initialize input, output, input gradient, weight gradient and bias gradient
            initialize_zeros(&this->x, this->in_features);
            initialize_zeros(&this->fx, this->out_features);
            initialize_zeros(&this->dfx, this->in_features);
            initialize_zeros(&this->dW, this->in_features * this->out_features);
            if (this->bias_flag) {
                initialize_zeros(&this->db, this->out_features);
            }
        }

        // Print layer information
        void show() {
            std::printf("%s: %d -> %d\n", this->name, this->in_features, this->out_features);
        }

        

        // Forward call
        double* forward(const double *input) {
            // Copy input to x
            memcpy(this->x, input, this->in_features * sizeof(double));

            // Compute linear transformation
            this->linear_transformation();

            // Return output
            return this->fx;
        }

        // Backward call
        double* backward(const double *upstream_grad) {
            // Compute linear transformation gradient
            this->linear_transformation_grad(upstream_grad);

            // Return output gradient
            return this->dfx;
        }

        // Update parameters
        void update_params(const double lr) {
            // Update weights
            for (int i = 0; i < this->in_features; i++) {
                for (int j = 0; j < this->out_features; j++) {
                    this->weight[i * this->out_features + j] -= lr * this->dW[i * this->out_features + j];
                }
            }

            // Update bias
            if (this->bias_flag) {
                for (int i = 0; i < this->out_features; i++) {
                    this->bias[i] -= lr * this->db[i];
                }
            }
        }
        
    
    private:
        // Linear transformation - fx(1, out) = W(in x out) x x(1, out) + b (1, out)
        void linear_transformation() {
            // Compute linear transformation
            for (int i = 0; i < this->out_features; i++) {

                // Reset fx for each output neuron
                this->fx[i] = 0.0;

                // Get the dot product of weight and input
                for (int j = 0; j < this->in_features; j++) {
                    this->fx[i] += this->weight[i * this->in_features + j] * this->x[j];
                }

                // Add bias
                if (this->bias_flag) {
                    this->fx[i] += this->bias[i];
                }
            }
        }

        // Linear transformation gradient - dfx(1, in) = dfx(1, out) x W(out x in)
        void linear_transformation_grad(const double *upstream_grad) {
            // Compute linear transformation gradient
            // dfx(1, in) = dfx(1, out) x W(out x in)
            for (int i = 0; i < this->in_features; i++) {
                // Reset dfx for each input neuron
                this->dfx[i] = 0.0;

                // Get the dot product of upstream gradient and weight
                for (int j = 0; j < this->out_features; j++) {
                    this->dfx[i] += upstream_grad[j] * this->weight[i * this->out_features + j];
                }
            }

            // Compute weight gradient
            // dW(in x out) = x(1, in) x dfx(1, out)
            // self.dW = (np.matmul(self.x[:, :, None], upstream_grad[:, None, :])).mean(axis=0)
            for (int i = 0; i < this->in_features; i++) {
                for (int j = 0; j < this->out_features; j++) {
                    this->dW[i * this->out_features + j] = this->x[i] * upstream_grad[j];
                }
            } 


            // Compute bias gradient
            if (this->bias_flag) {
                // db(1, out) = dfx(1, out)
                memcpy(this->db, upstream_grad, this->out_features * sizeof(double));
            }
        }
};
#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
    public:
        // Initialize the Tensor
        int n_batches;
        int n_features;
        double *data;

        // Initialize the Tensor
        Tensor(const int n_batches, const int n_features, const double *data);

        // Destructor
        ~Tensor();

        // Print tensor information
        void show();
};

#endif
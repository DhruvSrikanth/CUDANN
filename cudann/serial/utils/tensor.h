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

        // Print tensor data
        void print();

        // Clip tensor data
        void clip(const double min, const double max);
};

void free_tensor(Tensor *tensor);
void copy_tensor(Tensor *dst, Tensor *src);
void create_tensor(Tensor *tensor, const int n_batches, const int n_features);
void initialize_tensor(Tensor *tensor, const int n_batches, const int n_features, const double *data);

#endif
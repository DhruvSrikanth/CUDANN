#ifndef TENSOR_H
#define TENSOR_H


class Tensor {
    public:
        // Initialize the Tensor
        int batch_size;
        int n_features;
        double *data;

        // Initialize the Tensor
        Tensor(const int batch_size, const int n_features, const double *data);

        // Destructor
        ~Tensor();

        // Print tensor information
        void show();

        // Print tensor data
        void print();

        // Clip tensor data
        void clip(const double* min, const double* max);

        // Check tensor data for NaN
        bool has_nan();

        // Get the argmax of the tensor
        void argmax(int *b_argmax);

        // Sum tensor data
        double sum();
};

void free_tensor(Tensor *tensor);
void copy_tensor(Tensor *dst, const Tensor *src);
void create_tensor(Tensor *tensor, const int batch_size, const int n_features);
void initialize_tensor(Tensor *tensor, const int batch_size, const int n_features, const double *data);
void print_array(const double *data, const int batch_size, const int n_features);
bool check_array_for_nan(const double *array, const int size);

#endif
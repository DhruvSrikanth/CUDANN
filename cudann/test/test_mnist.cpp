#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "../serial/cudann.h"

struct MiniBatch {
    Tensor *input;
    Tensor *target;
};

struct Dataloader {
    MiniBatch *minibatches;
    int n_batches;
};

struct Subset {
    int channels;
    int image_size;
    unsigned char** images;
    unsigned char* labels;
    int n_samples;
};

struct Dataset {
    Subset train;
    Subset test;
};

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

unsigned char* read_mnist_labels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void train(const int n_classes, const int n_features, const double learning_rate, const int epochs, Dataloader *dataloader, NN *model, CrossEntropy *criterion) {
    // Print model summary
    model->summary();

    printf("Loss function: ");
    criterion->show();

    Tensor *input = (Tensor*) malloc(sizeof(Tensor));
    Tensor *target = (Tensor*) malloc(sizeof(Tensor));

    printf("Training model for %d epochs with learning rate %f.\n", epochs, learning_rate);
    // Train the model
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Compute average loss over a batch
        double avg_loss = 0.0;
        float progress = 0.0;
        int barWidth = 70;
        for (int mb = 0; mb < dataloader->n_batches; mb++) {
            std::cout << "Epoch: " << epoch + 1 << " - [";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) {
                    std::cout << "=";
                } else if (i == pos) {
                    std::cout << ">";
                } else {
                    std::cout << " ";
                }
            }
            std::cout << "] - " << mb << "/" << dataloader->n_batches << " (" << int(progress * 100.0) << "%) " << "Average Loss: " << (double) avg_loss / mb << "\r";
            std::cout.flush();

            progress += (float) 1.0 / dataloader->n_batches;
            
            // Get minibatch and copy it to the appropriate device
            copy_tensor(input, dataloader->minibatches[mb].input);
            copy_tensor(target, dataloader->minibatches[mb].target);

            // Forward pass
            Tensor *output = model->forward(input);

            // Compute loss
            Tensor *loss = criterion->forward(output, target);
            
            // Compute the loss gradient
            Tensor *downstream_grad = criterion->backward();

            // Backward pass
            model->backward(downstream_grad);

            // Update weights
            model->update_weights(learning_rate);

            // Compute average loss
            avg_loss += loss->sum() / loss->batch_size;
        }
        avg_loss /= dataloader->n_batches;

        std::cout << std::endl;
    }
}

int main() {
    const int batch_size = 64;
    const int n_classes = 10;
    const int n_features = 28*28;
    const double learning_rate = 0.01;
    const int epochs = 5;

    // Load MNIST dataset
    // Training set
    Subset train_set;
    train_set.channels = 1;
    train_set.images = read_mnist_images("cudann/test/data/mnist/train-images-idx3-ubyte", train_set.n_samples, train_set.image_size);
    train_set.labels = read_mnist_labels("cudann/test/data/mnist/train-labels-idx1-ubyte", train_set.n_samples);

    // Test set
    Subset test_set;
    test_set.channels = 1;
    test_set.images = read_mnist_images("cudann/test/data/mnist/t10k-images-idx3-ubyte", test_set.n_samples, test_set.image_size);
    test_set.labels = read_mnist_labels("cudann/test/data/mnist/t10k-labels-idx1-ubyte", test_set.n_samples);

    // Create dataset
    Dataset dataset;
    dataset.train = train_set;
    dataset.test = test_set;

    // Dataset properties
    std::cout <<"==============================" << std::endl;
    std::cout << "Training set: " << dataset.train.n_samples << " samples" << std::endl;
    std::cout << "Test set: " << dataset.test.n_samples << " samples" << std::endl;
    std::cout << "Image size: " << dataset.train.image_size << std::endl;
    std::cout << "Channels: " << dataset.train.channels << std::endl;
    std::cout <<"==============================" << std::endl;

    // Create minibatches
    const int n_train_batches = (dataset.train.n_samples / batch_size) + 1;

    // Create dataloader
    Dataloader train_dataloader;
    train_dataloader.minibatches = (MiniBatch*) malloc(n_train_batches * sizeof(MiniBatch));
    train_dataloader.n_batches = n_train_batches;

    // Get dataloader
    int sample_index;
    for (int mb = 0; mb < n_train_batches; mb++) {
        // Create minibatch
        double *input = (double*) malloc(batch_size * dataset.train.image_size * dataset.train.channels * sizeof(double));
        double *target = (double*) malloc(batch_size * n_classes * sizeof(double));
        
        for (int s = 0; s < batch_size; s++) {
            sample_index = mb * batch_size + s;
            if (sample_index >= dataset.train.n_samples) {
                break;
            }
            for (int p = 0; p < dataset.train.image_size * dataset.train.channels; p++) {
                // Image data is already normalized
                input[s * dataset.train.image_size * dataset.train.channels + p] = dataset.train.images[sample_index][p];
            }
            // Binarize labels
            for (int c = 0; c < n_classes; c++) {
                if (dataset.train.labels[sample_index] == c) {
                    target[s * n_classes + c] = 1.0;
                } else {
                    target[s * n_classes + c] = 0.0;
                }
            }
            
        }
        
        // Create tensor
        Tensor input_tensor(batch_size, dataset.train.image_size, input);
        Tensor target_tensor(batch_size, n_classes, target);
        
        // Create minibatch
        MiniBatch minibatch;
        minibatch.input = (Tensor*) malloc(sizeof(Tensor));
        minibatch.target = (Tensor*) malloc(sizeof(Tensor));
        copy_tensor(minibatch.input, &input_tensor);
        copy_tensor(minibatch.target, &target_tensor);
        
        // Add minibatch to dataloader 
        train_dataloader.minibatches[mb] = minibatch;
    }  

    // Create test dataloader
    const int n_test_batches = (dataset.test.n_samples / batch_size) + 1;

    // Create dataloader
    Dataloader test_dataloader;
    test_dataloader.minibatches = (MiniBatch*) malloc(n_test_batches * sizeof(MiniBatch));
    test_dataloader.n_batches = n_test_batches;

    // Get dataloader
    for (int mb = 0; mb < n_test_batches; mb++) {
        // Create minibatch
        double *input = (double*) malloc(batch_size * dataset.test.image_size * dataset.test.channels * sizeof(double));
        double *target = (double*) malloc(batch_size * n_classes * sizeof(double));
        
        for (int s = 0; s < batch_size; s++) {
            sample_index = mb * batch_size + s;
            if (sample_index >= dataset.test.n_samples) {
                break;
            }
            for (int p = 0; p < dataset.test.image_size * dataset.test.channels; p++) {
                // Image data is already normalized
                input[s * dataset.test.image_size * dataset.test.channels + p] = dataset.test.images[sample_index][p];
            }
            // Binarize labels
            for (int c = 0; c < n_classes; c++) {
                if (dataset.test.labels[sample_index] == c) {
                    target[s * n_classes + c] = 1.0;
                } else {
                    target[s * n_classes + c] = 0.0;
                }
            }
            
        }
        
        // Create tensor
        Tensor input_tensor(batch_size, dataset.test.image_size, input);
        Tensor target_tensor(batch_size, n_classes, target);
        
        // Create minibatch
        MiniBatch minibatch;
        minibatch.input = (Tensor*) malloc(sizeof(Tensor));
        minibatch.target = (Tensor*) malloc(sizeof(Tensor));
        copy_tensor(minibatch.input, &input_tensor);
        copy_tensor(minibatch.target, &target_tensor);
        
        // Add minibatch to dataloader 
        test_dataloader.minibatches[mb] = minibatch;
    }



    // Create Model
    // Create layers
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

    // Define objective
    // Get loss function
    CrossEntropy criterion("cross_entropy");

    // Train model
    train(n_classes, n_features, learning_rate, epochs, &test_dataloader, &model, &criterion);


    return 0;
}
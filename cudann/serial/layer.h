#ifndef LAYER_H
#define LAYER_H

class Layer {
    public:
        Layer();
        ~Layer();
        virtual void show();
        virtual double* forward(const double *input);
        virtual double* backward(const double *upstream_grad);
        virtual void update_params(const double learning_rate);
};

#endif
#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer route_layer;

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size);
route_layer make_route_layer_backend(int batch, int n, int *input_layers, int *input_size, BACKEND backend);
void forward_route_layer(const route_layer l, network net);
void forward_route_layer_vectorized(const route_layer l, network net);
void backward_route_layer(const route_layer l, network net);
void resize_route_layer(route_layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net);
void backward_route_layer_gpu(const route_layer l, network net);
#endif

#endif

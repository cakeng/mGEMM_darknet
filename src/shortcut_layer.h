#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
layer make_shortcut_layer_backend(int batch, int index, int w, int h, int c, int w2, int h2, int c2, BACKEND backend, int vectorized, int vecsize, int devectorize);
void forward_shortcut_layer(const layer l, network net);
void forward_shortcut_layer_backend(const layer l, network net, BACKEND backend);
void backward_shortcut_layer(const layer l, network net);
void resize_shortcut_layer(layer *l, int w, int h);

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net);
void backward_shortcut_layer_gpu(const layer l, network net);
#endif

#endif

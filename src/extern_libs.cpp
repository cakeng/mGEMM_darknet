#include <iostream>
#include <cstring>
#include <fstream> 
#include <bitset>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>
#include <cblas.h>
#include "extern_libs.h"
#include "ptmm.h"
extern "C"
{
    #include "activations.h"
    #include "batchnorm_layer.h"
    #include "convolutional_layer.h"
    #include "blas.h"
    #include "im2col.h"
}

float *NCHWtoNHWC(float *input, int blocks, int channels, int height, int width)
{
    float *out;
    if (posix_memalign((void **)(&out), 128, blocks * channels * height * width * sizeof(float)))
    {
        printf("Test input NHWC - POSIX memalign failed.");
    }
    for (int bIdx = 0; bIdx < blocks; bIdx++)
    {
        for (int cIdx = 0; cIdx < channels; cIdx++)
        {
            float *saveTarget = out + bIdx * channels * height * width + cIdx;
            float *loadTarget = input + bIdx * channels * height * width + cIdx * height * width;
            for (int hIdx = 0; hIdx < height; hIdx++)
            {
                for (int wIdx = 0; wIdx < width; wIdx++)
                {
                    *(saveTarget) = *(loadTarget);
                    saveTarget += channels;
                    loadTarget += 1;
                }
            }
        }
    }
    return out;
}

float* deVectorizeToNHWC (float* input, int blocks, int channels, int height, int width, int vecSize)
{
    float* out;
    if(posix_memalign((void**)(&out), 16, blocks * channels * height * width*sizeof(float)))
    {
        printf ("Devectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val;
                    val = *(input + b*channels*height*width + ((c/vecSize)*height*width + h*width +  w)*vecSize + (c%vecSize));
                    *(out + b*(height*width*channels) + h*(width*channels) + w*channels + c) = val;
                }
            }
        }
    }
    return out;
}

float* deVectorizeToNCHW (float* input, int blocks, int channels, int height, int width, int vecSize)
{
    float* out;
    if(posix_memalign((void**)(&out), 16, blocks * channels * height * width*sizeof(float)))
    {
        printf ("Devectorize - POSIX memalign failed.");
    }
    for (int b = 0; b < blocks; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0 ; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val;
                    val = *(input + b*channels*height*width + ((c/vecSize)*height*width + h*width +  w)*vecSize + (c%vecSize));
                    *(out + b*(height*width*channels) + c*height*width + h*width + w) = val;
                }
            }
        }
    }
    return out;
}

void deVectorize (layer l, int toNHWC)
{
    #ifdef __GEMMPLUS_DEBUG
    printf("GEMMPLUS Devectorizing...\n");
    printf("Batch: %d, Out_c: %d, Out_h: %d, Out_w: %d, Vecsize: %d\n"
        , l.batch, l.out_c, l.out_h, l.out_w, l.vecsize);
    #endif
    float* tmp;
    if (!toNHWC)
    {
        tmp = deVectorizeToNCHW(l.output, l.batch, l.out_c, l.out_h, l.out_w, l.vecsize);
    }
    else
    {
        tmp = deVectorizeToNHWC(l.output, l.batch, l.out_c, l.out_h, l.out_w, l.vecsize);
    }
    memcpy(l.output, tmp, sizeof(float) * l.outputs * l.batch);
    free(tmp);
}

void gemmplus_convolutional_layer(convolutional_layer l, network net)
{
    ptmm::ptmm_num_threads = std::thread::hardware_concurrency();
    // #ifdef __GEMMPLUS_DEBUG
    // printf("GEMMPLUS Wrapper Called.\n");
    // printf("Batch: %d, n: %d, Out_c: %d, Out_h: %d, Out_w: %d, In_c: %d, Depthwise: %d, Vecsize: %d\n"
    //     , l.batch, l.n, l.out_c, l.out_h, l.out_w, l.c, l.groups != 1, l.vecsize);
    // #endif
    ptmm ptmmFilter(0, l.weights, l.out_c, l.c, l.size, l.size, false, l.groups != 1);
    ptmmFilter.conv(net.input, l.output, nullptr, l.batch, l.h, l.w, l.pad, l.stride, 1);
    if(l.batch_normalize)
    {
        forward_batchnorm_layer_vectorized(l, net, l.vecsize);
    }
    else 
    {
        add_bias_vectorized(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w, l.vecsize);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);

    if (l.devectorize)
    {
        deVectorize (l, 0);
    }
}

void armnn_convolutional_layer(convolutional_layer l, network net)
{
    printf("ARMNN Wrapper Called.\n");
}

void xnnpack_convolutional_layer(convolutional_layer l, network net)
{
    printf("XNNPACK Wrapper Called.\n");
}

void direct_convolutional_layer(convolutional_layer l, network net)
{
    printf("DIRECT Wrapper Called.\n");
}

void openblas_convolutional_layer(convolutional_layer l, network net)
{
    // #ifdef __GEMMPLUS_DEBUG
    // printf("OPENBLAS Wrapper Called.\n");
    // #endif
        int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;

            im2col_cpu(net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w,
                l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
    activate_array(l.output, l.outputs*l.batch, l.activation);
}
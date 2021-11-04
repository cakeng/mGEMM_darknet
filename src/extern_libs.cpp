#include <iostream>
#include <cstring>
#include <fstream> 
#include <bitset>
#include <chrono>
#include <thread>
#include <iomanip>
#include <random>
#include <cblas.h>
#include <xnnpack.h>
#include <math.h>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>
#include <arm_compute/runtime/Scheduler.h>
#include "extern_libs.h"
#include "ptmm.h"
extern "C"
{
    #include <sys/time.h>
    #include "activations.h"
    #include "batchnorm_layer.h"
    #include "convolutional_layer.h"
    #include "blas.h"
    #include "im2col.h"
}

float *NCHWtoNHWC(float *input, int blocks, int channels, int height, int width)
{
    float *out = nullptr;
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
    float* out = nullptr;
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
    float* out = nullptr;
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
    float* tmp;
    #ifdef __GEMMPLUS_DEBUG
    // printf("Devectorizing...\n");
    // printf("Batch: %d, Out_c: %d, Out_h: %d, Out_w: %d, vectorized: %d, vecsize: %d\n"
        // , l.batch, l.out_c, l.out_h, l.out_w, l.vectorized, l.vecsize);
    #endif
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
    clock_t time=clock();
    ptmm::ptmm_num_threads = std::thread::hardware_concurrency();
    // #ifdef __GEMMPLUS_DEBUG
    // printf("GEMMPLUS Wrapper Called.\n");
    // printf("Batch: %d, n: %d, Out_c: %d, Out_h: %d, Out_w: %d, In_c: %d, Depthwise: %d, Vecsize: %d\n"
    //     , l.batch, l.n, l.out_c, l.out_h, l.out_w, l.c, l.groups != 1, l.vecsize);
    // #endif
    ptmm ptmmFilter(0, l.weights, l.out_c, l.c, l.size, l.size, false, l.groups != 1);

    time=clock();    ptmmFilter.conv(net.input, l.output, l.biases, l.batch, l.h, l.w, l.pad, l.stride, 1);
    printf("%f,", sec(clock()-time));

    time=clock();
    activate_array(l.output, l.outputs*l.batch, l.activation);
    if (l.devectorize)
    {
        deVectorize (l, 0);
    }
    printf("%f,", sec(clock()-time));
}

void armnn_convolutional_layer(convolutional_layer l, network net)
{
    // using namespace armnn;
    // // Load graph into runtime
    // run->LoadNetwork(networkIdentifier, std::move(optNet));
    // armnn::InputTensors inputTensors{{0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), input)}};
    // armnn::OutputTensors armnnOutputTensors{{0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), armnnOutput)}};

    // // Execute network
    // run->EnqueueWorkload(networkIdentifier, inputTensors, armnnOutputTensors);
}
void make_armnn_layer(convolutional_layer* l)
{
    // using namespace armnn;

    // arm_compute::Scheduler::get().set_num_threads(threadNum);

    // // Construct ArmNN network
    // armnn::NetworkId networkIdentifier;
    // INetworkPtr myNetwork = INetwork::Create();

    // armnn::Convolution2dDescriptor Conv2dDesc;
    // Conv2dDesc.m_PadTop = padding;
    // Conv2dDesc.m_PadBottom = padding;
    // Conv2dDesc.m_PadRight = padding;
    // Conv2dDesc.m_PadLeft = padding;
    // Conv2dDesc.m_StrideX = stride;
    // Conv2dDesc.m_StrideY = stride;
    // Conv2dDesc.m_DilationX = dilation;
    // Conv2dDesc.m_DilationY = dilation;
    // Conv2dDesc.m_BiasEnabled = true;
    // Conv2dDesc.m_DataLayout = DataLayout::NCHW;

    // armnn::DepthwiseConvolution2dDescriptor ConvDepth2dDesc;
    // ConvDepth2dDesc.m_PadTop = padding;
    // ConvDepth2dDesc.m_PadBottom = padding;
    // ConvDepth2dDesc.m_PadRight = padding;
    // ConvDepth2dDesc.m_PadLeft = padding;
    // ConvDepth2dDesc.m_StrideX = stride;
    // ConvDepth2dDesc.m_StrideY = stride;
    // ConvDepth2dDesc.m_DilationX = dilation;
    // ConvDepth2dDesc.m_DilationY = dilation;
    // ConvDepth2dDesc.m_BiasEnabled = true;
    // ConvDepth2dDesc.m_DataLayout = DataLayout::NCHW;

    // TensorInfo weightsInfo;
    // TensorInfo inputTensorInfo;
    // TensorInfo outputTensorInfo;
    // TensorInfo biasInfo(TensorShape({testBlocks}), DataType::Float32);

    // bool del = false;
    // float *filter;
    // float *input;

    // // For NHWC
    // if (depthwise == 0)
    // {
    //     weightsInfo = TensorInfo(TensorShape({testBlocks, testFilHeight, testFilWidth, testChannels}), DataType::Float32);
    //     filter = testFilterTensorNHWC;
    //     Conv2dDesc.m_DataLayout = DataLayout::NHWC;
    // }
    // else
    // {
    //     weightsInfo = TensorInfo(TensorShape({1, testChannels, testFilHeight, testFilWidth}), DataType::Float32);
    //     filter = testFilterTensorNHWC;
    //     ConvDepth2dDesc.m_DataLayout = DataLayout::NHWC;
    // }
    // input = testInputTensorNHWC;
    // inputTensorInfo = TensorInfo(TensorShape({testBatch, testHeight, testWidth, testChannels}), DataType::Float32);
    // outputTensorInfo = TensorInfo(TensorShape({testBatch, testHout, testWout, testBlocks}), DataType::Float32);

    // armnn::ConstTensor bias(biasInfo, testBiasTensor);
    // armnn::ConstTensor weights(weightsInfo, filter);
    // IConnectableLayer *convLayer;
    // if (depthwise == 0)
    // {
    //     convLayer = myNetwork->AddConvolution2dLayer(Conv2dDesc, weights, bias, "Armnn Conv2d");
    // }
    // else
    // {
    //     convLayer = myNetwork->AddDepthwiseConvolution2dLayer(ConvDepth2dDesc, weights, bias, "Armnn Conv2d Depthwise");
    // }
    // IConnectableLayer *InputLayer = myNetwork->AddInputLayer(0);
    // IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    // InputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    // convLayer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    // // Create ArmNN runtime
    // IRuntime::CreationOptions options; // default options
    // IRuntimePtr run = IRuntime::Create(options);

    // //Set the tensors in the network.
    // InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    // convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // // Optimise ArmNN network
    // armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {Compute::CpuAcc}, run->GetDeviceSpec());
    // if (!optNet)
    // {
    //     // This shouldn't happen for this simple sample, with reference backend.
    //     // But in general usage Optimize could fail if the hardware at runtime cannot
    //     // support the model that has been provided.
    //     std::cerr << "Error: Failed to optimise the input network." << std::endl;
    // }
}

void xnnpack_convolutional_layer(convolutional_layer l, network net)
{
    // printf("XNNPACK Wrapper Called.\n");
    // printf("Batch: %d, n: %d, Out_c: %d, Out_h: %d, Out_w: %d, In_c: %d, Groups: %d, Vecsize: %d\n"
    //     , l.batch, l.n, l.out_c, l.out_h, l.out_w, l.c, l.groups, l.vecsize);
    clock_t time=clock();
    if (net.xnnpack_threadpool != l.xnnpack_pthreadpool)
    {
        std::cerr << "XNNPACK threadpool assertion failed." << std::endl;
    }
    xnn_status status;
    
    if (l.prev_output != net.input)
    {
        std::cerr << "XNNPACK Prev output == input failed." << std::endl;
        
        status = xnn_setup_convolution2d_nhwc_f32(
        (xnn_operator_t)l.xnnpack_op,
        l.batch /* batch size */, l.h /* input height */, l.w /* input width */,
        net.input /* input */, l.output /* xnnpackOutput */,
        (pthreadpool_t)net.xnnpack_threadpool /* threadpool */);
        if (status != xnn_status_success)
        {
            std::cerr << "failed to setup operation #0" << std::endl;
        }
    }

    xnn_run_operator((xnn_operator_t)l.xnnpack_op, (pthreadpool_t)net.xnnpack_threadpool);
    printf("%f,", sec(clock()-time));

    time=clock();
    activate_array(l.output, l.outputs*l.batch, l.activation);
    if (l.devectorize)
    {
        deVectorize (l, 0);
    }
    printf("%f,", sec(clock()-time));
    
}

void make_xnnpack_layer(convolutional_layer* l)
{
    xnn_status status;
    pthreadpool_t threadpool = (pthreadpool_t)l->xnnpack_pthreadpool;

    if (xnn_initialize(nullptr /* allocator */) != xnn_status_success)
    {
        std::cerr << "failed to initialize XNNPACK" << std::endl;
    }
    xnn_operator_t op = nullptr;
    status = xnn_create_convolution2d_nhwc_f32(
        l->pad /* top padding */, l->pad /* right padding */,
        l->pad /* bottom padding */, l->pad /* left padding */,
        l->size /* kernel height */, l->size /* kernel width */,
        l->stride /* subsampling height */, l->stride /* subsampling width */,
        1 /* dilation_height */, 1 /* dilation_width */,
        l->groups /* groups */,
        l->c/l->groups /* input channels per group */,
        l->out_c/l->groups /* xnnpackOutput_channels_per_group */,
        l->c /* input pixel stride */,
        l->out_c /* xnnpackOutput pixel stride */,
        l->weights, l->biases,
        -(__builtin_inff()) /* xnnpackOutput min */, (__builtin_inff()) /* xnnpackOutput max */,
        0 /* flags */,
        &op);
    if (status != xnn_status_success)
    {
        std::cerr << "failed to create operation #0 - status: " << status << std::endl;
    }
    if (l->prev_output != NULL)
    {
        status = xnn_setup_convolution2d_nhwc_f32(
        op,
        l->batch /* batch size */, l->h /* input height */, l->w /* input width */,
        l->prev_output /* input */, l->output /* xnnpackOutput */,
        threadpool /* threadpool */);
        if (status != xnn_status_success)
        {
            std::cerr << "failed to setup operation #0" << std::endl;
        }
    }
    l->xnnpack_op = op;
    // printf("XNNPACK Conv layer created. - addr %lld\n", (uint64_t)l->xnnpack_op);
    // printf("Batch: %d, n: %d, Out_c: %d, Out_h: %d, Out_w: %d, In_c: %d, Groups: %d, Vecsize: %d\n"
    //     , l->batch, l->n, l->out_c, l->out_h, l->out_w, l->c, l->groups, l->vecsize);
}

void* xnnpack_pthreadpool_create()
{
    return (void*)pthreadpool_create(std::thread::hardware_concurrency());
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

    add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);

    activate_array(l.output, l.outputs*l.batch, l.activation);
}
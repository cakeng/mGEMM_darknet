Modified Darknet code used to test the performance of mGEMM: Low-latency Convolution with Minimal Memory Overhead Optimized for Mobile Devices [[Paper Link](https://dl.acm.org/doi/10.1145/3498361.3538940)]

mGEMM is referred to as "PTMM" or "GEMM PLUS" in the code. These are just the development names of mGEMM.


1. Use the makefile to build the program. Please use ARMv8 & Aarch64 (64bit OS) based environments. Raspberry Pi 4 with 64-bit Raspberry Pi OS is known to work.

2. Use the command "wget https://pjreddie.com/media/files/yolov3-tiny.weights" to download the filter weights for YoloV3-Tiny network.

3. Use the command "./darknet detector backend cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg <CONVOLUTION_BACKEND>" to run the YoloV3-Tiny with different convolution backend.
<CONVOLUTION_BACKEND> can be either GEMMPLUS, ARMNN, XNNPACK, DEFAULT, OPENBLAS, depending on your choice. GEMMPLUS refers to the mGEMM algorithm backend.


.a files for XNNPACK and ARMNN versions we used are included in the git. ARMNN library is modified to use the GEMM-based convolution method. 

You'll have to install OPENBLAS on your own. "apt install libopenblas-dev" usually works. Please refer to the Makefile for other dependencies.

Original Darknet Repo: https://github.com/pjreddie/darknet

Source code for mGEMM: https://github.com/cakeng/GEMM_Plus


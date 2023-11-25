# Execution framework for mGEMM: Low-latency Convolution with Minimal Memory Overhead Optimized for Mobile Devices in MobiSys'22

This is the modified [Darknet framework](https://github.com/pjreddie/darknet) used in the evaluation of MobiSys'22 paper **mGEMM: Low-latency Convolution with Minimal Memory Overhead Optimized for Mobile Devices** [[Paper Link](https://dl.acm.org/doi/10.1145/3498361.3538940)]

Below is the teaser for mGEMM that I made for the MobiSys'22 conference. (Click the image to watch it on YouTube!)

[![Youtube Link](http://img.youtube.com/vi/iKwQjuxQYhw/0.jpg)](https://youtu.be/iKwQjuxQYhw)

You can download additional materials (Posters, Presentation Slides, Graphics) used in this project from my [[Homepage](https://www.cakeng.info/home)].

---

## Usage Instructions

mGEMM, amongst other convolution solutions from XNNPACK, ARMNN, etc., is included in this repository and can be selected as a drop-in replacement of the baseline convolution kernel from the Darknet framework.

1. Use the makefile to build the program. **Please use ARMv8 & Aarch64 (64bit OS) based environments.** Raspberry Pi 4 with 64-bit Raspberry Pi OS is known to work.

2. Use the command "wget https://pjreddie.com/media/files/yolov3-tiny.weights" to download the filter weights for the YoloV3-Tiny network.

3. Use the command "./darknet detector backend cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg <CONVOLUTION_BACKEND>" to run the YoloV3-Tiny with different convolution backend.
<CONVOLUTION_BACKEND> can be either **GEMMPLUS**, **ARMNN**, **XNNPACK**, **DEFAULT**, or **OPENBLAS**, depending on your choice. **GEMMPLUS** refers to the mGEMM algorithm.

**mGEMM is referred to as "PTMM" or "GEMM PLUS" in the code. These are just the development names of mGEMM.**

Libraries (.a files) for mGEMM, XNNPACK, and ARMNN versions we used are included in the git. ARMNN library is modified to use the GEMM-based convolution method. 

You'll have to install OPENBLAS on your own. "apt install libopenblas-dev" usually works. Please refer to the Makefile for other dependencies.

Please reach out to me at "cakeng at snu dot ac dot kr" if you have any questions!

---

## Source code Links

The source code for the mGEMM library can be found at [https://github.com/cakeng/GEMM_Plus](https://github.com/cakeng/GEMM_Plus)

The source code for the ARMNN library can be found at [https://github.com/ARM-software/armnn](https://github.com/ARM-software/armnn).

The source code for the XNNPACK library can be found at [https://github.com/google/XNNPACK](https://github.com/google/XNNPACK).

Original Darknet Repo can be found at [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)


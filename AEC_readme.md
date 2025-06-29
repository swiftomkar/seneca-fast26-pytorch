## Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca

This repository contains the source code implementation of the FAST'26 paper "Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca".

Input data preprocessing is a common bottleneck when concurrently training multimedia machine learning (ML) models in modern systems. To alleviate these bottlenecks and reduce the training time for concurrent jobs, we present Seneca, a data loading system that optimizes cache partitioning and data sam- pling for the data storage and ingestion (DSI) pipeline. The design of Seneca contains two key techniques. First, Seneca uses a performance model for the data pipeline to optimally partition the cache for three different forms of data (encoded, decoded, and augmented). Second, Seneca opportunistically serves cached data over uncached ones during random batch sampling so that concurrent jobs benefit from each other. We implement Seneca by modifying PyTorch and demonstrate its effectiveness by comparing it against several state-of-the-art caching systems for DNN training. Seneca reduces the makespan by 45.23% compared to PyTorch and increases data processing throughput by up to 3.45× compared to the next best dataloader.

## Source code
The implementation for this project is split across 2 repositories:
1) https://github.com/swiftomkar/seneca-fast26-pytorch
2) https://github.com/swiftomkar/seneca-fast26-torchvision
While these repositories hold the source code, we also provide a docker image that you can download and use to run and evaluate Seneca.
For AEC members: For ease and efficiency of evaluation, you can directly start with the docker container setup.

## Setup

Our evaluations were completed on CloudLab, Azure, and our internal servers. We recognise that it is currently challenging to get access to GPU VMs and to help with that, we can provide access to our internal servers for evaluation purposes. Please send an email to odesai@syr.edu and we will be happy to provide the necessary access. 

To run Seneca you will need a NVIDIA GPU (tested on A100, V100 and RTX5000) with CUDA 11.7, GPU driver version 417.56, docker, and Python 3. We used the prebuilt NVIDIA docker container nvcr.io/nvidia/pytorch:19.05-py3 container as the base image and built a Seneca container image on top of it. The image can be downloaded by:
```
docker pull omkarbdesai/seneca_cuda11.7_cudnn8.5:v2.2
```
You will only need to make sure that you have the right NVIDIA drivers installed on your system. The rest of the setup will be provided by the docker container automatically. 

## Data
Our experiments use the following publicly available large datasets, which can be downloaded from their official repos.

ImageNet-1k (~146GB): This is the most widely used image dataset (with 1000 classes) downloadable from here

OpenImages : We use the extended version of OpenImages with 4260 classes (~645GB), which includes the 600 class OpenImages v4 dataset here, along with the crowdsourced image subset here

ImageNet-22k (~1.3TB) : This is the full ImageNet dataset with 21841 classes downloadable from the ImageNet website.

For the purposes of validating the functionality of Seneca, you only need a small dataset. For this purpose, you can use the test images from the imagenet dataset for classification tasks.
![Screenshot 2025-06-29 at 11 26 38 AM](https://github.com/user-attachments/assets/bb2af2be-870c-4ad8-b4d9-2ae013bce9a8)

```
https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
```

## Running Seneca

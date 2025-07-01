## Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca

This repository contains the source code implementation of the FAST'26 paper "Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca".

Input data preprocessing is a common bottleneck when concurrently training multimedia machine learning (ML) models in modern systems. To alleviate these bottlenecks and reduce the training time for concurrent jobs, we present Seneca, a data loading system that optimizes cache partitioning and data sam- pling for the data storage and ingestion (DSI) pipeline. The design of Seneca contains two key techniques. First, Seneca uses a performance model for the data pipeline to optimally partition the cache for three different forms of data (encoded, decoded, and augmented). Second, Seneca opportunistically serves cached data over uncached ones during random batch sampling so that concurrent jobs benefit from each other. We implement Seneca by modifying PyTorch and demonstrate its effectiveness by comparing it against several state-of-the-art caching systems for DNN training. Seneca reduces the makespan by 45.23% compared to PyTorch and increases data processing throughput by up to 3.45× compared to the next best dataloader.

## Process to verify "Artifact available"

### Source code
The implementation for this project is split across 2 repositories:
1) https://github.com/swiftomkar/seneca-fast26-pytorch
2) https://github.com/swiftomkar/seneca-fast26-torchvision


While these repositories hold the source code, we also provide a docker image that you can download and use to run and evaluate Seneca.

## Process to verify "Artifact functional"

### Setup

Our evaluations were completed on CloudLab, Azure, and our internal servers. We recognise that it is currently challenging to get access to GPU VMs and to help with that, we can provide access to our internal servers for evaluation purposes. Please send an email to odesai@syr.edu and we will be happy to provide the necessary access. 

To run Seneca you will need a NVIDIA GPU (tested on A100, V100 and RTX5000) with CUDA 11.7, a GPU driver version compatible with your GPU, docker, and Python 3. We used the prebuilt NVIDIA docker container nvcr.io/nvidia/pytorch:19.05-py3 container as the base image and built a Seneca container image on top of it. The image can be downloaded by:

```
docker pull omkarbdesai/seneca_cuda11.7_cudnn8.5:v2.2
```
You will only need to make sure that you have the right NVIDIA drivers and the appropriate docker runtime installed on your system. The rest of the setup will be provided by the docker container automatically. 

The instructions to installing nvidia drivers can be found [here](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/). 

The instructions to installing the NVIDIA Container Toolkit can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit)

### Data
Our experiments use the following publicly available large datasets, which can be downloaded from their official repos.

ImageNet-1k (~146GB): This is the most widely used image dataset (with 1000 classes) downloadable from [here](https://image-net.org/download-images.php)

OpenImages : We use the extended version of OpenImages with 4260 classes (~645GB), which includes the 600 class OpenImages v4 dataset here, along with the crowdsourced image subset [here](https://storage.googleapis.com/openimages/web/index.html)

ImageNet-22k (~1.3TB) : This is the full ImageNet dataset with 21841 classes downloadable from the ImageNet website [here](https://image-net.org/download-images.php).

For the purposes of validating the functionality of Seneca, you can download and extract the smallest dataset which is the ~146GB ImageNet-1K dataset.

```
https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
```

### Redis setup

Seneca's implementation uses redis as the in-memory cache to store and retrieve training data. While redis is currently used, it can be swapped for other in-memory databases.
You will need to run 4 redis instances using the followng commands. Redis instances can be run on the same server or on a remote caching node or a cluster of nodes.

#### Redis installation

```
cd $HOME
wget -P $HOME/ http://download.redis.io/releases/redis-6.0.1.tar.gz 
tar xzf redis-6.0.1.tar.gz
mv $HOME/redis-6.0.1 $HOME/redis-stable
cd $HOME/redis-stable
yes Y | sudo apt-get install tcl
make
```



#### Running redis
To run all necessary redis instances with the default port configurations, run the following script where redis is installed.

```
redis_utils/start_redis.sh
```

This will launch all necessary redis instances on the same node. 

To manage KV pair evictions from the cache, run the following script in the background:

```
redis_utils/cache_eviction_handler.py
```

### System initialization

To run the Docker container, you can use the following command example

```
sudo docker run --gpus all -it --rm -v <host path to dataset>:<container path to map dataset> -v /dev/shm/:/dev/shm omkarbdesai/seneca_cuda11.7_cudnn8.5:v2.2
```

#### Running Seneca

Once inside the docker container, run the following example command to run training of a ResNet50 model using Seneca:

```
python -m torch.distributed.launch --nproc_per_node=<number of available GPUs per node> --master_port 1234 pytorch-imagenet-mp.py --crop_size=224 -a vit_b_16 -b 256 --workers 16 --noeval --node_rank <node number, 0 indexed> --epochs 5 --job_sample_tracker_port 6388 --raw_cache_port 6378 --tensor_cache_port 6380 --decoded_cache_port 6376 --decoded_cache_host 10.56.82.137 --raw_cache_host 10.56.82.137 --tensor_cache_host 10.56.82.137 --amp --no_dali --ImageFolder BBModel --cache_allocation <cache size in GB> --cache_sllit 0-0-100 --classes 1000 <path to dataset containing the "train" directory>
```

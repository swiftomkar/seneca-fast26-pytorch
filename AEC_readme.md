# Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca

This repository contains the source code implementation of the FAST'26 paper "Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca".

Input data preprocessing is a common bottleneck when concurrently training multimedia machine learning (ML) models in modern systems. To alleviate these bottlenecks and reduce the training time for concurrent jobs, we present Seneca, a data loading system that optimizes cache partitioning and data sam- pling for the data storage and ingestion (DSI) pipeline. The design of Seneca contains two key techniques. First, Seneca uses a performance model for the data pipeline to optimally partition the cache for three different forms of data (encoded, decoded, and augmented). Second, Seneca opportunistically serves cached data over uncached ones during random batch sampling so that concurrent jobs benefit from each other. We implement Seneca by modifying PyTorch and demonstrate its effectiveness by comparing it against several state-of-the-art caching systems for DNN training. Seneca reduces the makespan by 45.23% compared to PyTorch and increases data processing throughput by up to 3.45× compared to the next best dataloader.

# Seneca: Artifact Evaluation Instructions

This document provides step-by-step instructions to verify the **Artifact Available** and **Artifact Functional** badges for the *Seneca* artifact (FAST '26).

---

## ✅ Artifact Available

### 🔗 Source Code Repositories

Seneca's implementation is divided across two repositories:

1. [Seneca Core (PyTorch)](https://github.com/swiftomkar/seneca-fast26-pytorch)
2. [Seneca TorchVision Extensions](https://github.com/swiftomkar/seneca-fast26-torchvision)

For convenience, we also provide a prebuilt Docker image to run and evaluate Seneca:

```bash
docker pull omkarbdesai/seneca_cuda11.7_cudnn8.5:v3.0
````

---

## ⚙️ Artifact Functional

Follow the steps below to set up and run Seneca.

---

## Step 1: 🖥️ System Requirements

* **GPU**: NVIDIA (tested on A100, V100, RTX5000)
* **CUDA**: Version 11.7
* **Driver**: Compatible NVIDIA driver
* **Python**: Version 3.x
* **Docker**: Installed and GPU-enabled

Useful setup links:

* [Install NVIDIA GPU Drivers](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)
* [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Step 2: 📦 Dataset Setup

Seneca supports the following public datasets:

| Dataset                         | Classes | Size    | Link                                                                       |
| ------------------------------- | ------- | ------- | -------------------------------------------------------------------------- |
| **ImageNet-1K**                 | 1,000   | \~142GB | [Download](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) |
| **OpenImages (v4 + extension)** | 600     | \~517GB | [OpenImages v4](https://storage.googleapis.com/openimages/web/index.html)  |
| **ImageNet-22K**                | 21,841  | \~1.4TB | [Download](https://image-net.org/download-images.php)                      |

For validation purposes, we recommend starting with **ImageNet-1K**.

---

## Step 3: 🧠 Redis Setup

Seneca uses **Redis** as in-memory caches for different data formats.

### ⚠️ Recommended Setup

* **Ideally, install Redis on the host machine or a dedicated cache node**, **not** inside the Docker container.

---

### 3.1 Install Redis (on host or cache node)

```bash
cd $HOME
wget http://download.redis.io/releases/redis-6.0.1.tar.gz
tar xzf redis-6.0.1.tar.gz
mv redis-6.0.1 redis-stable
cd redis-stable
yes Y | sudo apt-get install tcl
make
sudo apt install redis-tools
```

---

### 3.2 Modify Redis Configuration

Edit `~/redis-stable/redis.conf`:

```conf
bind 127.0.0.1 -> bind 0.0.0.0
protected_mode yes -> protected_mode no
```

These changes allow Redis to accept external connections (e.g., from the Docker container).

---

### 3.3 Launch Redis Instances

From the Redis install directory:

```bash
redis_utils/start_redis.sh
```

---

### 3.4 Start Cache Eviction Handler

On the host:

```bash
python redis_utils/cache_eviction_handler.py &
```

This script manages eviction of KV pairs.

---

## Step 4: 🐳 Launch the Docker Container

```bash
sudo docker run --gpus all -it --rm \
    -v <host_path_to_dataset>:<container_dataset_path> \
    -v /dev/shm/:/dev/shm \
    omkarbdesai/seneca_cuda11.7_cudnn8.5:v3.0
```

Replace `<host_path_to_dataset>` with the path where your dataset is stored.

---

## Step 5: 🔧 Inside the Docker Container if compute capability is other than sm_70 (V100 GPU):

### 5.1 Build PyTorch 

```bash
cd /workspace/disyml_seneca_pytorch
python setup.py clean
python setup.py develop
```

### 5.2 Build TorchVision

```bash
cd /workspace/vision
python setup.py clean
python setup.py develop
```

### 5.3 Install Apex

```bash
cd ~/apex_patched/apex

# For pip >= 23.1
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./

# For pip < 23.1
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
#   --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

---

## Step 6: 🧪 Run Training with Seneca

### 6.1 Start the final Redis instance inside the container

```bash
/workspace/redis-stable/src/redis-server /workspace/redis-stable/redis.conf --port 6377 &
```

### 6.2 Launch Seneca with this example command

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpus> --master_port 1234 \
    pytorch-imagenet-mp.py --crop_size=<crop size> -a resnet50 -b 256 --workers 16 --noeval \
    --node_rank <node_index> --epochs 5 --job_sample_tracker_port 6388 \
    --raw_cache_port 6378 --tensor_cache_port 6380 --decoded_cache_port 6376 \
    --decoded_cache_host <host_ip> --raw_cache_host <host_ip> --tensor_cache_host <host_ip> \
    --amp --no_dali --ImageFolder BBModel \
    --cache_allocation <cache_size_in_GB> --cache_sllit 0-90-10 --classes 1000 \
    <path_to_dataset_with_train_directory>
```

Replace placeholders:

* `<num_gpus>`: Number of GPUs per node
* `<node_index>`: Node rank (0-indexed)
* `<host_ip>`: Host machine IP running Redis
* `<cache_size_in_GB>`: Total cache size to allocate
* `<crop size>`: Size of image to train on (eg: 64, 128)
* `<path_to_dataset_with_train_directory>`: Path that contains the `train/` directory

---

## 🛠️ Troubleshooting

### 🚫 Redis Connection Refused

**Symptoms:**

* Errors like `Connection refused` when trying to reach Redis servers.

**Fixes:**

* Ensure Redis servers are listening on all interfaces (`bind 0.0.0.0`)
* Disable protected mode (`protected_mode no`)
* Make sure Docker container can reach the host IP. Try using `host.docker.internal` on macOS/Windows or the host’s internal IP on Linux.
* Verify Redis ports (6376, 6377, 6378, 6380, 6388) are open.

### ❗ CUDA Compatibility Errors

If you encounter errors related to CUDA architecture:

```bash
export TORCH_CUDA_ARCH_LIST="<your_GPU_compute_capability>"  # e.g., "8.0"
export USE_CUDNN=1
export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
export CUDNN_LIBRARY=/usr/local/cuda/lib64

cd /workspace/disyml_seneca_pytorch
python setup.py clean
python setup.py develop

cd /workspace/vision
python setup.py clean
python setup.py develop
```

---

## 📩 Need Help?

If you do not have access to compatible GPU infrastructure, we can provide temporary access to our internal evaluation servers.
Please contact: **[odesai@syr.edu](mailto:odesai@syr.edu)**

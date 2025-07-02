
# Seneca: Artifact Evaluation Instructions

This guide provides step-by-step instructions to verify the *Artifact Available* and *Artifact Functional* badges for our Seneca FAST '26 artifact.

---

## ✅ Artifact Available

### 🔗 Source Code

Seneca is implemented across two public GitHub repositories:

1. [Seneca PyTorch Core](https://github.com/swiftomkar/seneca-fast26-pytorch)
2. [Seneca TorchVision Extensions](https://github.com/swiftomkar/seneca-fast26-torchvision)

You may choose to clone these repositories and build from source, but we also provide a ready-to-use Docker image for ease of use.

---

## ⚙️ Artifact Functional

The following steps will walk you through setting up, configuring, and running the Seneca system.

---

## Step 1: 🖥️ Hardware and Software Requirements

- **GPU**: NVIDIA GPU (tested on A100, V100, RTX5000)
- **CUDA**: Version 11.7
- **Driver**: Compatible NVIDIA driver
- **Python**: Version 3.x
- **Docker**: Installed and configured for GPU

📦 Pull the Docker container:
```bash
docker pull omkarbdesai/seneca_cuda11.7_cudnn8.5:v2.2
````

🔧 Additional setup help:

* [Install NVIDIA Drivers](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)
* [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Step 2: 📦 Dataset Setup

Seneca supports multiple datasets. For functional verification, use ImageNet-1k (\~146GB):

🔗 Download from:
[https://image-net.org/challenges/LSVRC/2012/2012-downloads.php](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)

Extract the dataset locally. You will mount this into the container in Step 5.

---

## Step 3: 🧠 Redis Cache Setup

Seneca uses Redis as an in-memory cache backend.

### 3.1 Install Redis

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

### 3.2 Launch Redis Instances

Run this inside the container or from the host if Redis is external:

```bash
redis_utils/start_redis.sh
```

If Redis is run outside the container, update `redis.conf`:

```conf
bind 127.0.0.1 -> bind 0.0.0.0
protected_mode yes -> protected_mode no
```

### 3.3 Start Cache Eviction Handler

```bash
python redis_utils/cache_eviction_handler.py &
```

---

## Step 4: 🚀 Start the Docker Container

```bash
sudo docker run --gpus all -it --rm \
    -v <path-to-dataset-on-host>:<path-in-container> \
    -v /dev/shm/:/dev/shm \
    omkarbdesai/seneca_cuda11.7_cudnn8.5:v2.2
```

Replace `<path-to-dataset-on-host>` with the actual dataset path.

---

## Step 5: 🔧 Install Python Dependencies (Inside Container)

If your GPU has a compute capability other than `sm_70`, you must rebuild PyTorch and TorchVision.

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
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./
```

If `pip >= 23.1` is not supported:

```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

---

## Step 6: 🧪 Run Training with Seneca

### 6.1 Start Redis (if not running)

```bash
/workspace/redis-stable/src/redis-server /workspace/redis-stable/redis.conf --port 6377 &
```

### 6.2 Launch Training

```bash
python -m torch.distributed.launch --nproc_per_node=<num-gpus> --master_port 1234 \
    pytorch-imagenet-mp.py --crop_size=224 -a resnet50 -b 256 --workers 16 --noeval \
    --node_rank <node-index> --epochs 5 --job_sample_tracker_port 6388 \
    --raw_cache_port 6378 --tensor_cache_port 6380 --decoded_cache_port 6376 \
    --decoded_cache_host <host-ip> --raw_cache_host <host-ip> --tensor_cache_host <host-ip> \
    --amp --no_dali --ImageFolder BBModel \
    --cache_allocation <cache-size-GB> --cache_sllit 0-0-100 --classes 1000 \
    <path-to-dataset-with-train-dir>
```

Replace:

* `<num-gpus>`: Number of GPUs on this node
* `<node-index>`: Node number (starting at 0)
* `<host-ip>`: Redis host IP (use `localhost` if on the same node)
* `<cache-size-GB>`: Cache size (e.g., 64)
* `<path-to-dataset-with-train-dir>`: Path to folder containing `train/`

---

## 🛠 Troubleshooting

### ❗ CUDA Capability Mismatch

If you see CUDA-related errors:

```bash
export TORCH_CUDA_ARCH_LIST="<your GPU capability>"  # e.g., "8.0"
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

## 💬 Support

If you're unable to access compatible GPU infrastructure, email us at **[odesai@syr.edu](mailto:odesai@syr.edu)**. We can provide temporary access to internal servers for review purposes.

---

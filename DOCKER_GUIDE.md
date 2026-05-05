# Docker Setup Guide for MAL-ZDA Framework

## 📦 What was created

- **Dockerfile** - Standard CPU-based Docker image
- **Dockerfile.gpu** - GPU-optimized Docker image with CUDA
- **docker-compose.yml** - Container orchestration with optional GPU support
- **requirements.txt** - Python dependencies
- **.dockerignore** - Files to exclude from Docker image

---

## 🚀 Quick Start

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually included with Docker Desktop)
- For GPU support: NVIDIA GPU + NVIDIA Docker runtime

---

## 🔧 Building the Docker Image

### CPU Version (Recommended for Development)
```bash
docker build -t mal-zda:latest .
```

### GPU Version
```bash
docker build -f Dockerfile.gpu -t mal-zda:gpu .
```

---

## ▶️ Running the Container

### Using Docker Compose (Easiest)

#### CPU Version
```bash
docker-compose up
```

#### Interactive Shell (CPU)
```bash
docker-compose run --rm mal-zda bash
```

#### GPU Version
```bash
docker-compose up mal-zda-gpu
```

---

### Using Docker CLI

#### Run with Default Command (Tests)
```bash
docker run --rm mal-zda:latest
```

#### Interactive Bash Shell
```bash
docker run -it --rm mal-zda:latest bash
```

#### With Dataset Volume
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
  mal-zda:latest bash
```

#### Run Specific Script
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
  mal-zda:latest python mal_zda_framework.py
```

#### GPU Support
```bash
docker run -it --rm \
  --gpus all \
  -v "$(pwd)/dataset:/app/dataset" \
  mal-zda:gpu python mal_zda_framework.py
```

---

## 📁 Volume Mounts

The container expects these directories for data persistence:

```
./dataset/        # Input CSV files
./outputs/        # Generated outputs and checkpoints
./results/        # Results and analysis
```

**Mount syntax:**
```bash
-v "/host/path:/container/path"
```

**Examples:**
```bash
# Mount single directory
docker run -v "$(pwd)/dataset:/app/dataset" mal-zda:latest

# Mount multiple directories
docker run \
  -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/results:/app/results" \
  mal-zda:latest
```

---

## 🔍 Checking if Everything Works

### Inside Container
```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check all dependencies
python -c "import torch, numpy, pandas, sklearn; print('All packages loaded successfully!')"

# Run tests
python test_mal_zda_framework.py
```

---

## 📊 Using the Framework in Container

### Example: Run the Framework
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
  mal-zda:latest \
  python mal_zda_framework.py
```

### Example: Interactive Session
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/app/dataset" \
  mal-zda:latest \
  bash

# Inside container:
# python
# >>> from mal_zda_framework import MALZDAFramework
# >>> framework = MALZDAFramework()
# >>> ...
```

---

## 🐳 Docker Compose Services

The `docker-compose.yml` defines two services:

### 1. CPU Service (`mal-zda`)
```bash
docker-compose up mal-zda          # Run with default command
docker-compose run mal-zda bash    # Interactive shell
```

### 2. GPU Service (`mal-zda-gpu`)
```bash
docker-compose up mal-zda-gpu      # Requires GPU + nvidia-docker
```

---

## 🛠️ Customization

### Modify Base Image
Edit `Dockerfile` or `Dockerfile.gpu`:
```dockerfile
# Change Python version
FROM python:3.10-slim

# Or use different PyTorch version
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
```

### Add System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y \
    your-package-name \
    && rm -rf /var/lib/apt/lists/*
```

### Change Default Command
```dockerfile
# Modify the CMD at the end
CMD ["python", "your_script.py"]
```

---

## 📈 Performance Tips

### CPU Optimization
```bash
docker run --rm \
  --cpus=4 \
  --memory=8g \
  mal-zda:latest
```

### GPU Optimization
```bash
docker run --rm \
  --gpus all \
  --cpus=4 \
  --memory=16g \
  mal-zda:gpu
```

---

## 🐛 Troubleshooting

### Port Issues
If Docker can't build, ensure port 8888 is not in use:
```bash
docker ps -a  # List all containers
docker rm container-name  # Remove conflicting container
```

### Permission Issues
```bash
# Run with user permissions
docker run -u $(id -u):$(id -g) mal-zda:latest

# Or fix volume ownership
docker run -it mal-zda:latest chown -R $(id -u):$(id -g) /app
```

### GPU Not Detected
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.6.2-runtime-ubuntu20.04 nvidia-smi

# If not installed, follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory
Reduce batch size in configuration:
```bash
# Inside container
docker run --memory=4g mal-zda:latest python mal_zda_framework.py
```

---

## 🚢 Deployment

### Push to Registry
```bash
# Tag image
docker tag mal-zda:latest your-registry.com/mal-zda:latest

# Push to Docker Hub or private registry
docker push your-registry.com/mal-zda:latest
```

### Deploy with Docker Compose on Server
```bash
scp docker-compose.yml user@server:/path/to/project
ssh user@server
cd /path/to/project
docker-compose up -d
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [NVIDIA Docker Runtime](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)

---

## ✅ Next Steps

1. Build the image: `docker build -t mal-zda:latest .`
2. Prepare your dataset in `./dataset/` directory
3. Run the container: `docker run -v "$(pwd)/dataset:/app/dataset" -v "$(pwd)/outputs:/app/outputs" mal-zda:latest`
4. Check outputs in `./outputs/` directory

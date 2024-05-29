Here's a polished version of your README:

---

# FLAMEGPU2 Documentation and Support

For comprehensive documentation and support, please visit the official [FLAMEGPU2 GitHub repository](https://github.com/FLAMEGPU/FLAMEGPU2).

## Using the DockerFile

### Step 1: Install Docker
To get started with Docker, follow the installation instructions on the [Docker website](https://www.docker.com/get-started/).

### Step 2: Enable GPU Compatibility for Docker
Ensure your Docker setup supports GPU by following the installation guide for NVIDIA's container toolkit [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Step 3: Build and Run the Docker Image

1. **Build the Docker Image:**
    ```bash
    sudo docker build -t flamegpu .
    ```

2. **Run the Docker Container:**
    ```bash
    sudo docker run --runtime=nvidia --gpus all flamegpu
    ```

---



## Documentation and Support
Please refer to the official flameGPU2 git at https://github.com/FLAMEGPU/FLAMEGPU2

## Use of the DockerFile
Install docker:
https://www.docker.com/get-started/

Install GPU compatibility for docker:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

run the docker image:
sudo docker build -t flamegpu .
sudo docker run --runtime=nvidia --gpus all flamegpu

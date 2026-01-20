# csdl_dafoam: CSDL - DAFoam interface and tutorial script for 2D airfoil aerodynamic shape optimization

**csdl_dafoam** is a CSDL-DAFoam interface to implement aerodynamic shape optimization. It integrates **lsdo_geo** for geometric parameterization, **IDWarp** for mesh deformation, **modopt** as the optimizer, **CSDL** for automatic differentiation, and **DAFoam** for primal and adjoint computations. 

<!--
### Distribution Statement
**Distribution Statement A**: Approved for public release; distribution is unlimited. PA\# AFRL-2025-3820.
-->

## Instructions to run the script for users
To run the script, first pull the Docker image and run the script as follows:

(1) Pull the Docker images from Docker Hub
```bash
docker pull cfdkang/csdl_dafoam
```

(2) Check the pulled Docker image
```bash
docker images
```

(3) Run (Mount) the Docker image
```bash
docker run -it --rm -u dafoamuser --mount \
"type=bind,src=$(pwd),target=/home/dafoamuser/mount" \
-w /home/dafoamuser/mount cfdkang/csdl_dafoam:latest bash
```

(4) You are ready to run the tutorial script
```bash
./run.sh
```

## Instructions to run the script for developers 
To further develop the CSDL/DAFoam interface, it is recommended to install all dependencies from scratch. For example, please refer to the DAFoam installation guide to set up the required prerequisites: https://dafoam.github.io/mydoc_installation_source.html#prerequisites. Developers can also add additional package installations in Dockerfile and build a new image to run the script build upon the existing one (https://github.com/cfdkang/Dockerfile/blob/main/Dockerfile).

(1) Download the Dockerfile:
```bash
https://github.com/cfdkang/Dockerfile/blob/main/Dockerfile
```

(2) Build the Dockerfile
```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -t csdl_dafoam:latest .
```

(3) Check the pulled Docker image
```bash
docker images
```

(4) Run (Mount) the Docker image
```bash
docker run -it --rm -u dafoamuser --mount \
"type=bind,src=$(pwd),target=/home/dafoamuser/mount" \
-w /home/dafoamuser/mount dafoam/csdl_dafoam:latest bash
```

(5) You are ready to run the tutorial script
```bash
./run.sh
```

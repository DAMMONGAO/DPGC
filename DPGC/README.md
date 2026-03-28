# DPGC 

## Setup and Installation
The code was tested on Ubuntu 20/22 and Cuda 11/12.</br>

Clone the repo
```
git clone https://github.com/princeton-vl/DPGC.git --recursive
cd DPGC
```
Create and activate the dpgc anaconda environment
```
conda env create -f environment.yml
conda activate dpgc
```

Next install package
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

pip install .

# download models and data (~2GB)
./download_models_and_data.sh
```

### Recommended - Install the Pangolin Viewer
Note: You will need to have CUDA 11 and CuDNN installed on your system.

1. Step 1: Install Pangolin (need the custom version included with the repo)
```
./Pangolin/scripts/install_prerequisites.sh recommended
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
cd ../..
```

2. Step 2: Install the viewer
```bash
pip install ./DPViewer
```

For installation issues, [Docker Image](https://github.com/princeton-vl/DPVO_Docker) supports the visualizer.

## Checkpoints
The pretrained models can be downloaded from ###DPGC/checkpoints###

## Evaluation
For example, we provide evaluation scripts for TartanAir and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=1 --split=validation --plot --save_trajectory 
```
To verify stability, you can run this five times and take the median of the results.

### ICL-NUIM:
```
python evaluate_icl_nuim.py --trials=1 --plot --save_trajectory
```
To verify stability, you can run this five times and take the median of the results.

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

If single gpu, you can run:
```
CUDA_VISIBLE_DEVICES=xxx python train_single_gpu.py --steps=240000 --lr=0.00008 --name=<your name>
```

If multi gpus, you can run:
```
CUDA_VISIBLE_DEVICES=xxx,yyy python train_single_gpu.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Acknowledgements
Thanks for DPVO(2022) and its code. Our code is built upon DPVO


#!/bin/bash
# This script is called to config a server with conda alread installed

# installing pytorch
yes | conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# installing pycoccotools
yes | conda install -c conda-forge pycocotools 
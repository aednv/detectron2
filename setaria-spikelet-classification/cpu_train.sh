#!/bin/bash

#BSUB -J train_model
#BSUB -q long
#BSUB -n 1
#BSUB -R "rusage[mem=20000]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[rh=8]"
#BSUB -W 48:00
#BSUB -o training.out
#BSUB -e training.err

singularity exec ../../singularity-container-images/detectron2-cpu-docker.simg python3 train_model.py


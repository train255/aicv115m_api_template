#!/bin/bash
mkdir -p audio_upload
mkdir -p meata_upload

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES='0'

python3 serve.py

#!/bin/bash
mkdir -p audio_upload
mkdir -p meta_upload

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES='0'

python3 serve.py

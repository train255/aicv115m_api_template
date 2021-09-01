#!/bin/bash

# Download train data
pip install gdown
mkdir -p data
cd data
gdown https://drive.google.com/uc?id=1Oq9UgA9cEGMNRGvF7oNKkFOg6udsDprl
gdown https://drive.google.com/uc?id=1UcRDOkq9cHfRrNOlFSaqJPNfCZPoFth_
# Download private test data
gdown https://drive.google.com/uc?id=1KJmH12giVJl8mIP-sTP2RK7Y5fKABFx_

# Unzip files
unzip -q aicv115m_final_public_train.zip -d ./
unzip -q aicv115m_extra_public_1235samples.zip -d ./
unzip -q aicv115m_final_private_test.zip -d ./

# Remove zip file
rm -f aicv115m_final_public_train.zip
rm -f aicv115m_extra_public_1235samples.zip
rm -f aicv115m_final_private_test.zip

cd ..

# Training
export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES='0'

python3 main.py train

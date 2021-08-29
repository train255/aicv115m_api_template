#!/bin/bash

pip install gdown
cd data
# Download train data
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
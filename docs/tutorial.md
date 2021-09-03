
## Train data
### 1 - Download train data
```bash
chmod +x scripts/download_trainset.sh
./scripts/download_trainset.sh
```

### 2 - Training
```bash
chmod +x scripts/run_train.sh
./scripts/run_train.sh
```

## Submission

### 1 - Download test data
```bash
chmod +x scripts/download_testset.sh
./scripts/download_testset.sh
```

### 2 - Create submission
You can download [the pretrained-model](https://drive.google.com/file/d/1zkXetgBifefAfQgS3o2j8zYlCGZi3S7h/view?usp=sharing) to create submission without training phase

```bash
chmod +x scripts/run_submission.sh
./scripts/run_submission.sh
```

## Reference
[High accuracy classification of COVID-19 coughs using Mel-frequency cepstral coefficients and a Convolutional Neural Network with a use case for smart home devices](https://www.researchsquare.com/article/rs-63796/v1.pdf?c=1598480611000)

[Fixed sound threshold level (librosa.effects.split)](https://mmchiou.gitbooks.io/ai_gc_methodology_2018_v1-private/content/zhong-wen-yu-yin-sentence-segmentation/acoustic-domain-sentence-segmentation/using-librosa-library.html)

[Sound Augmentation Librosa](https://www.kaggle.com/huseinzol05/sound-augmentation-librosa)
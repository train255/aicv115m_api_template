
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
You can download [the pretrained-model](https://drive.google.com/file/d/1EGq-fal8ty4n6HolOOgSKYhk3nraVM8_/view?usp=sharing) to create submission without training phase

```bash
chmod +x scripts/run_submission.sh
./scripts/run_submission.sh
```

## Run service
```bash
chmod +x scripts/run_service.sh
./scripts/run_service.sh
```
Test service
```python
import requests
import json
import time

url = 'http://localhost:8000/api/predict/'
headers = {
    'accept': 'application/json'
}

uuid = "123456789" # Insert user id here.
audio_path = "./data/audio.wav" # Insert file path of audio file here.
gender = None # Insert gender here.
age = None # Insert age here.
cough_type = None # Insert type of cough here.
health_status = None # Insert health status here.
note = None # Insert note here
audio_name="data.wav"

metadata = json.dumps(
    {
        "uuid": uuid,
        "subject_gender" : gender,
        "subject_age" : age,
        "subject_cough_type": cough_type,
        "subject_health_status": health_status,
        "note": note
    }
)

files = {
    'meta': (None, metadata),
    'audio_file': (audio_name, open(audio_path, 'rb')),
}

response = requests.post(url, headers=headers, files=files).json()
print(response)
```


## Reference
[High accuracy classification of COVID-19 coughs using Mel-frequency cepstral coefficients and a Convolutional Neural Network with a use case for smart home devices](https://www.researchsquare.com/article/rs-63796/v1.pdf?c=1598480611000)

[Fixed sound threshold level (librosa.effects.split)](https://mmchiou.gitbooks.io/ai_gc_methodology_2018_v1-private/content/zhong-wen-yu-yin-sentence-segmentation/acoustic-domain-sentence-segmentation/using-librosa-library.html)

[Sound Augmentation Librosa](https://www.kaggle.com/huseinzol05/sound-augmentation-librosa)
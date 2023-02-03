#!/bin/bash
### install requirements for layoutLMv2

pip install numpy==1.24.1
pip install torchvision==0.9.0+cu101 torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.25.1
pip install editdistance==0.6.2
pip install wandb==0.13.9
pip install pillow==8.1.0
pip install einops==0.6.0
pip install omegaconf==2.3.0
pip install datasets==2.9.0
pip install seaborn==0.12.2
apt-get update
apt-get -y install libgl1-mesa-glx
pip install opencv-python==4.7.0.68
pip install multiprocess==0.70.13 dill==0.3.5.1
apt install tesseract-ocr
pip install pytesseract==0.3.10
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
pip install openpyxl
pip install jupyter
pip install ipykernel
pip install nltk==3.8.1

python3 -c "
import nltk
nltk.download('averaged_perceptron_tagger')
"
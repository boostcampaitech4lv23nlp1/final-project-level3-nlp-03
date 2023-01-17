#!/bin/bash
### install requirements for layoutLMv2

pip install numpy==1.24.1
pip install torchvision==0.9.0+cu101 torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.25.1
pip install editdistance==0.6.2
pip install wandb==0.13.9
pip install pillow==8.1.0
pip install nltk
apt install tesseract-ocr
pip install pytesseract
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

python3
import nltk
nltk.download('averaged_perceptron_tagger')
exit()
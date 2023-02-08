# 1. Overview

![image](https://user-images.githubusercontent.com/73874591/217573079-2303b321-6c82-46a0-a0ef-b63cbaff4e00.png)
## Introduce
DocVQA(Document Visual Question Answering)은 RRC(Robust Reading Competetion)에서 2021년 내놓은 Task 중 하나로 기존의 DAR방식보다 한단계 더 높은 난이도의 TASK이다.

정확히는 문서 이미지의 텍스트(수기, 타이핑 또는 인쇄) 내용을 추출하고 해석할 뿐만 아니라 레이아웃(페이지 구조, 양식, 표), 텍스트가 아닌 요소(마크, 체크박스, 구분자, 다이어그램) 및 스타일(글꼴, 색상, 강조 표시)을 포함한 수많은 다른 시각적 단서를 활용하는 TASK이다.

## What we done?
1. 데이터셋에 제공되어있지 않은 Answer index를 찾기 위해 기존의 Hit algorithm을 유클리드기반으로 수정
2. Category별 Data Annotation, Error Analysis
3. Visualize Attention Heatmap
4. Decoder Generate

## 2. Project Tree
```
DocVQA
├─ configs
│  └─ baseline.yaml
├─ data_proces
│  └─ LayoutLMPreprocess.py
├─ install
│  └─ install_requirements.sh
├─ jupyter
│  ├─ Datasets.ipynb
│  ├─ inference.ipynb
│  └─ LayoutLMv2.ipynb
├─ model
│  ├─ BaselineModel.py
│  └─ Decoder.py
├─ save
│  └─ model.pt
├─ trainer
│  ├─ BaselineTrainer.py
│  └─ DecoderTrainer.py
├─ utils
│  ├─ check_dir.py
│  ├─ metric.py
│  ├─ seed_setting.py
│  └─ wandb_setting.py
├─ .gitignore
├─ git_convention.md
├─ train.py
├─ generate.py
└─ inference.py
```
## 3. Contributors
|김근형|김찬|유선종|이헌득|
|:---:|:---:|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/97590480/205299519-174ef1be-eed6-4752-9f3d-49b64de78bec.png">|<img src="https://user-images.githubusercontent.com/97590480/205299316-ea3dc16c-00ec-4c37-b801-3a75ae6f4ca2.png">|<img src="https://user-images.githubusercontent.com/97590480/205299037-aec039ea-f8d3-46c6-8c11-08c4c88e4c56.jpeg">|<img src="https://user-images.githubusercontent.com/97590480/205299457-5292caeb-22eb-49d2-a52e-6e69da593d6f.jpeg">|
|[Github](https://github.com/kimkeunhyeong)|[Github](https://github.com/chanmuzi)|[Github](https://github.com/Trailblazer-Yoo)|[Github](https://github.com/hundredeuk2)|
- 김근형: Deocder, Streamlit Demo, Fine-tuning
- 김찬: Result Analysis, Encoder, Question Maker Exp.
- 유선종: AttentionHeatmap, Hit Algorithm, Refactoring Code, Encoder
- 이헌득: Decoder, Baseline Modeling, BoundingBox Exp. Code Reviewer

## 4. Project Pipeline
![image](https://user-images.githubusercontent.com/73874591/217577054-e2be3421-f74b-42dd-9653-30ce812e75e7.png)

---------------------------------
## Reference
Mathew, M., Karatzas, D., & Jawahar, C. V. (2021). Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 2200-2209).
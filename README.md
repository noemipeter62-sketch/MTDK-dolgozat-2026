# MTDK-dolgozat-2026
Agydaganatok MRI-alapú osztályozása és szegmentálása Fourier Neural Operator kiterjesztéssel és PyQt6 felülettel.
##  Projekt leírás

Ez a projekt MRI felvételek feldolgozásával foglalkozik, ahol a cél:

-  daganatok osztályozása (classification)
-  daganatok szegmentálása (segmentation)
-  Fourier-alapú jellemzők beépítése neurális hálókba

A rendszer különböző deep learning modelleket használ, valamint frekvenciatérbeli (Fourier) transzformációkat alkalmaz a teljesítmény javítása érdekében.

## 🧠 Klasszifikáció (Classification)

### 📂 DenseNet121
- densenet121_classification.ipynb
- densenet121_fourier_classification.ipynb

### 📂 ResNet50
- resnet50_classification.ipynb
- resnet50_fourier_classification.ipynb

### 📂 EfficientNetB3
- efficientnetb3_classification.ipynb
- efficientnetb3_fourier_classification.ipynb

### 📂 Ensemble
- ensemble_classification.ipynb

## Szegmentálás (Segmentation)

### 📂 U-Net
- unet_fourier.ipynb
- unet_fourier_evaluation.ipynb

### 📂 Attention U-Net
- attention_unet_segmentation.ipynb
- attention_unet_evaluation.ipynb

### 📂 DeepLabV3+
- deeplabv3plus_segmentation.ipynb
- deeplabv3plus_fourier_segmentation.ipynb
- deeplabv3plus_fourier_evaluation.ipynb

### 📂 Ensemble
- ensemble_segmentation.ipynb
- ensemble_evaluation.ipynb

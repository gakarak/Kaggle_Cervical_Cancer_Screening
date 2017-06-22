# Challenge_Cervical_Cancer_Screening
Kaggle Challenge: Intel &amp; MobileODT Cervical Cancer Screening


This repository contains the program code and data for the Kaggle competition **Intel & MobileODT Cervical Cancer Screening**

To prepare preprocessed dataset and trained models, please download arhives with data & models after cloning this repo:


Preparation of trained models (this step is necessary to start the demo). Models for cervix&channel classifiacation is available in this repository [models/segm](models/segm), and models for cervix types classification is avaiulable in cloud-storage: [Trained on basic data](https://cloud.mail.ru/public/GMHD/yeteP2zmW) and [Trained on additional data](https://cloud.mail.ru/public/MA5K/w4KEndxsB):

```
cd Kaggle_Cervical_Cancer_Screening
cd models
wget https://cloud.mail.ru/public/GMHD/yeteP2zmW -O model_cervix_classification_basic.tar
tar xf model_cervix_classification_basic.tar

wget https://cloud.mail.ru/public/MA5K/w4KEndxsB -O model_cervix_classification_with_add.tar
tar xf model_cervix_classification_with_add.tar
```

To run Demo code go to the **demo** directory & run segmentation & classification tasks:

```
cd Kaggle_Cervical_Cancer_Screening
cd demo

./start01-automatic-segmention.sh             # to start segmentation task
./start02-automatic-classification-basic.sh   # to start classification task
./start03-automatic-classification-add.sh
```

Pre-processed data (for stage1) is available for download from: [Basic dataset for training](https://cloud.mail.ru/public/AqBM/dYhCgs4V5) and [Additional data](https://cloud.mail.ru/public/HYUS/PFwexrDuC)


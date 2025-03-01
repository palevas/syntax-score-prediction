# End-to-end SYNTAX score prediction: benchmark and methods

![Scheme](seq_model_scheme.png)

## Run inference:
```bash
python seq_apply.py -r DATASET_DIR -w WEIGHTS_DIR --fold 0
```
DATASET_DIR - dataset directory.\
Sample dataset: https://disk.yandex.com/d/drZKKBJnH2r8vg \
Full dataset: https://zenodo.org/records/14005818

WEIGHTS_DIR - model weights directory.\
Sample weights: https://disk.yandex.com/d/_4ARTacETFQr1A

## Train backbone:
```bash
mkdir backbone
# train backbone model for left coronary artery
python backbone_train.py -r DATASET_DIR -a left --fold 0
# train backbone model for right coronary artery
python backbone_train.py -r DATASET_DIR -a right --fold 0
```

## Train study-level sequence model
```bash
mkdir seq_models
# train sequence model for left coronary artery
python seq_train.py -r DATASET_DIR -a left --fold 0 --variant lstm_mean
# train sequence model for right coronary artery
python seq_train.py -r DATASET_DIR -a right --fold 0 --variant lstm_mean
```

## Dataset
This is the complete dataset that was used to train the model.\
CardioSYNTAX dataset: https://zenodo.org/records/14005818

### BibTeX reference
```
@InProceedings{Ponomarchuk_2025_WACV,
    author    = {Ponomarchuk, Alexander and Kruzhilov, Ivan and Mazanov, Gleb and Utegenov, Ruslan and Shadrin, Artem and Zubkova, Galina and Bessonov, Ivan and Blinov, Pavel},
    title     = {CardioSyntax: End-to-End SYNTAX Score Prediction - Dataset Benchmark and Method},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {5873-5883}
}
```

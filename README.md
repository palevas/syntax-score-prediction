# End-to-end SYNTAX score prediction: benchmark and methods

## Run inference:
```bash
python seq_apply.py -r DATASET_DIR -w WEIGHTS_DIR --fold 0
```
DATASET_DIR - dataset directory.\
Sample dataset: https://disk.yandex.com/d/drZKKBJnH2r8vg

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

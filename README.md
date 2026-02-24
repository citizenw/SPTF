# Scalable prediction of heterogeneous traffic flow with enhanced non-periodic feature modeling

## 1. Environment
Our experiments are conducted based on Python 3.8 and Pytorch 1.12.1.

And you can install other packages needed for experiments via:

```
pip install -r requirements.txt
```

## 2. Training
use the following command to train the model:

```
python experiments/train.py -c models/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'
```

Replace ${MODEL_NAME} and ${DATASET_NAME} with any supported models and datasets. 

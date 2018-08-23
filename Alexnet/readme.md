# VGG16

## Models
* alexnet_bn_after_relu.py 
* alexnet_bn.py -- Batch norm before ReLu
* ensemble -- Following https://arxiv.org/pdf/1801.09321.pdf


## Training Models
To train holistic model see "train_holistic.py"
```
python3 train_holistic.py -r True -rp ./alexnet.ckpt
```
* -r    "True": restore from flat -rp, or "False": train model from scratch
* -rp   ""Path to the checkpoint file to restore model from""


To train ensemble model see "train_ensemble.py"
```
python3 train_ensemble.py -r True -rp ./alexnet.ckpt
```
* -r    "True": restore entire model (sub-models included), or "False": restore all from pretrained holistic model
* -rp   ""Path to the checkpoint file to restore model from""


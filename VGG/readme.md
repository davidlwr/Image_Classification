# VGG16

## IMPORTANT / WARNING

Some saved models (not in the current git folder) use 3 channels, while some use 1 channel images
number of channels set for the graph can be set in the 'image_depth' param of the __init__ method of the specifc model. 


## Models
* VGG16
* VGG16_concat
* VGG16_selu
* VGG16_ensemble


## Training Models
To train holistic model see "train.py"

```
python3 train.py <FLAGS>
```

* -rp     Path to the checkpoint file to restore model from. leave blank to train from scratch
* -smg    "True": meta graph is saved along with variables. "False": Only variables are saved
* -s      Path to create tf.saved_model in, exits after model is created. Does not save if left blank

To train ensemble model see "train_ensemble.py"
```
python3 train_ensemble.py
```
* -r      "True": restore entire model (sub-models included), or "False": restore all from pretrained holistic model
* -rp     Path to the checkpoint file to restore model from
* smg     True: meta graph is saved along with variables. False: Only variables are saved


## Running saved model as RESTful API

See following:

'''
https://taiger.atlassian.net/wiki/spaces/INT/pages/463634433/Tensorflow+Distributed+-+REST+API
'''


## Regarding "VGG16_concat.py"
Attempting to increase inputs to FNN by concatenating outputs off softmax layers.

Currently, the GPU lacks memory to even handle the concatenation of the last 2 layers. 
* Maybe possible further in the future with better hardware.


## Regarding "VGG16_selu.py"
Testing ideas from:

* https://arxiv.org/abs/1706.02515
* https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9 

However tests results were less accurate than ReLu. Perhaps its effects can only be seen clearer on deeper networks
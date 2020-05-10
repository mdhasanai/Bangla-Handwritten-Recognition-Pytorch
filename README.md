## Bangla Handwritten Characters Recognition using Pytorch




### Requirements
- Python 3.5 or later
- Install Pytorch 1.0.0 or later (https://pytorch.org)
- Run ``❱❱❱  bash requirement.sh``

### Data
#### Bangla Handwritten Characters Recognition
You can download dataset from EkushNet (https://shahariarrabby.github.io/ekush/#home).
After complete download, extract the images into "data" folder
Make csv file for taining,validation and testing.
CSV file format should be:
```
/path/to/images1, class_number1
/path/to/images2, class_number2
```

### Model Architecture
#### The architecture of the model is illustrated by the following
Later, I will add the graph


### Training 
To train the the model, run the following command:

```python
❱❱❱ python train.py
```
To see the training progress in tensorboard, run the following command:
```python
❱❱❱ tensorboard --logdir="./runs/"
```
##### Note: To change or modify the model or parameters, see ``` config.py ``` file



#### Feel free to create an issue if you (the reader) come across any problems.

# SSD: Single Shot MultiBox Detector
```
├── src: Relevant modules to implement the SSD model    
│     ├── resnet50_backbone.py   Modify different backbones 
│     ├── ssd_model.py           SSD network structure file 
│     └── utils.py               Implementation of some functions used in the training process
├── train_utils: Training and validation related modules (including cocotools)  
├── my_dataset.py: Custom dataset is used to read the VOC dataset    
├── train_ssd300.py: Using different backbone as the backbone of the SSD network for training  
├── predict_test.py: Simple prediction script, using trained weights for prediction testing 
├── pascal_voc_classes.json: pascal_voc2012 label file
├── bdd100K_classes.json: bdd100k label file
├── plot_curve.py: Loss used to plot the training process and mAP for the validation set
└── validation.py: Use the COCO indicator of the trained weight validation/test data and generate the record_mAP.txt file
```
## Data set, using PASCAL VOC2012 and BDD100K data set (download and put it in the current folder of the project)
* Pascal VOC2012 train/val dataset download address：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* BDD100K：https://bdd-data.berkeley.edu/portal.html#download #Download 100K images and Detection 2020 Labels


## training method
* Make sure to prepare the dataset ahead of time
* Modify the custom data set and convert the corresponding file in the preprocess folder to a voc format data set
* Change the custom dataset name to VOC2022
* Use the corresponding backbone
* Single GPU training or CPU, directly use the train_ssd300.py training script
* The `results.txt` saved during the training process is the COCO indicator on the validation set for each epoch, the first 12 values are the COCO indicator, and the last two values are the training average loss and learning rate

## pre_trained weights
 
download from :https://github.com/psxzz13/SSD
 
vgg16_reducedfc: for SSD_VGG

nvidia_ssdpyt_fp32: for SSD_chaned_detection

# Grading Evaluation of Haploid Fertility Restoration Traits Based on Inception-ResNet in Maize

# Maize-IRNet
Maize-IRNet is a grading evaluation model of haploid anther emergence and ear seed setting based on Inception-ResNet.
Please refer to Release for specific APP download and use

# Maize-IRNet Introduction
Firstly, the modules of Stem and Inception-ResNet are utilized for image feature extraction and multi-scale feature learning. Then, the Reduction module is used for spatial downsampling and feature compression, and the global attention mechanism (GAM) is used to enhance the recognition of key regions of the image.

# Experimental Data Introduction
We constructed a dataset containing 1,897 high-resolution haploid ear images with different seed setting rates and 6,443 tassel images with different anther emergence rates.
For details, refer to [dataset](https://github.com/wyzwyz666/maize-haploid-fertility/blob/main/dataset)

# 1、Import dataset
```text
datasets1
└── traintest
    └── level0
        └── IMG_20220315_210823.jpg
```

# 2、Modify the  [cls_classes1.txt](https://github.com/wyzwyz666/maize-haploid-fertility/blob/main/sourcecode/model_data/cls_classes1.txt), where each line represents a category
```text
level0
level1
level2
level3
level4
level5
```
# 3、Create training.txt and test.txt files in the format of 
```text
0; datasets1/traintest/level0/IMG_20220317_095224.jpg
```
# 4、If you don't want to create it yourself, you can use the split_dataset_into_test_and_train_sets feature in train.py, which can automatically randomly divide the dataset proportionally
```text
# Retrieve the paths of all files and their category indexes
all_data_dir = 'datasets1/traintest'
train_files, test_files, train_labels, test_labels = split_dataset_into_test_and_train_sets(all_data_dir,test_size=0.1)
# Write the file paths and category indexes of the training and testing sets to a new txt file
with open('clstrain.txt', 'w') as f:
    for file, label in zip(train_files, train_labels):
        f.write(str(label) + ';' + file + '\n')

with open('clstest.txt', 'w') as f:
    for file, label in zip(test_files, test_labels):
        f.write(str(label) + ';' + file + '\n')
```
# 5、Environment
Install dependencies in [requirements.txt](https://github.com/wyzwyz666/maize-haploid-fertility/blob/main/requirements.txt)
```text
Pytorch installation can be done using this command
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
If some dependencies are missing, please install them yourself using the pip command
```
# 6、Adjust parameters，run train.py
```text
such as,
CUDA_DEVICES = 0, 1, 2, 3, 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
# 7、Questions
```text
If you have any further questions, please feel free to contact us：2023317110053@webmail.hzau.edu.cn
```

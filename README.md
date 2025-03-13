# 1、Import dataset
```text
datasets1
└── traintest
    └── level0
        └── IMG_20220315_210823.jpg
```

# 2、Modify the cls_classes1.txt in ./model_data/, where each line represents a category
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

# 5、Install dependencies in requirements.txt(If some dependencies are missing, please install them yourself using the pip command)


# 6、Adjustable parameters，run train.py

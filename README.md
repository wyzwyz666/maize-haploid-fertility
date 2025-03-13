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
```text
# # 获取所有文件的路径和它们的类别索引
# all_data_dir = 'datasets1/traintest'
# train_files, test_files, train_labels, test_labels = split_dataset_into_test_and_train_sets(all_data_dir,test_size=0.1)
# # 将训练集和测试集的文件路径和类别索引写入到新的txt文件中
# with open('clstrain0.txt', 'w') as f:
#     for file, label in zip(train_files, train_labels):
#         f.write(str(label) + ';' + file + '\n')

# with open('clstest0.txt', 'w') as f:
#     for file, label in zip(test_files, test_labels):
#         f.write(str(label) + ';' + file + '\n')
```
# 5、Install dependencies in requirements.txt(If some dependencies are missing, please install them yourself using the pip command)
```text
Pytorch installation can be done using this command
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
# 6、Adjust parameters，run train.py
```text
such as,
CUDA_DEVICES = 0, 1, 2, 3, 4, 5, 6, 7
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
# 7、If you have any further questions, please feel free to contact us：1319836632@qq.com

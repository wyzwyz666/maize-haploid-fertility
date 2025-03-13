# import dataset
datasets1

  └── traintest

      └── level0
    
          └── IMG_20220315_210823.jpg

# Modify the cls_classes1.txt file in model_data, where each line represents a category

# Create training. txt and test. txt files in the format of 
0; datasets1/traintest/level0/IMG_20220317_095224.jpg
# If you don't want to create it yourself, you can use the split_dataset_into_test_and_train_sets feature in train.py, which can automatically randomly divide the dataset proportionally
# Install dependencies in request.txt
# Adjustable parameters，run train.py

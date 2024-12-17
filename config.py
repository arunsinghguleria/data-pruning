unforgettable_example_path = '/data/home1/arunsg/gitproject/example_forgetting/array_single_line.txt'


nih_pathDirData = "/data/home1/arunsg/data-pruning/dataset/images"

nih_pathFileTrain = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_train.csv"
nih_pathFileVal = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_balanced-val.csv"
nih_pathFileBalancedTest = "/data/home1/arunsg/data-pruning/nih-cxr/nih-cxr-lt_single-label_balanced-test.csv"
model_storage = "/data/home1/arunsg/gitproject/data-pruning/base_model/"


prune_ratio = [0.7, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # here first value should be for class with highest number of examples, second for second hightest and so on...
prune_ratio = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # here first value should be for class with highest number of examples, second for second hightest and so on...
prune_ratio = [0.6930, 0.1015, 0.2038, 0.1413, 0.13261, 0.21591, 0.1555, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
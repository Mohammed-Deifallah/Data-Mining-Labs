from os import listdir
from os.path import isfile, join

def get_data(ROOT):
    images_dir = join(ROOT, 'images/')
    ground_truth_dir = join(ROOT, 'groundTruth/')
    train_dir = images_dir + 'train/'
    ground_truth_train = ground_truth_dir + 'train/'
    val_dir = images_dir + 'val/'
    ground_truth_val = ground_truth_dir + 'val/'
    test_dir = images_dir + 'test/'
    ground_truth_test = ground_truth_dir + 'test/'
    
    train_images = [join(train_dir + f) for f in listdir(train_dir) if isfile(join(train_dir, f)) and f.endswith(".jpg")]
    ground_truth_train_images = [join(ground_truth_train, f) for f in listdir(ground_truth_train) if isfile(join(ground_truth_train, f)) and f.endswith(".mat")]
    train_images.sort()
    ground_truth_train_images.sort()
     
    val_images = [join(val_dir, f) for f in listdir(val_dir) if isfile(join(val_dir, f)) and f.endswith(".jpg")]
    ground_truth_val_images = [join(ground_truth_val, f) for f in listdir(ground_truth_val) if isfile(join(ground_truth_val, f)) and f.endswith(".mat")]
    val_images.sort()
    ground_truth_val_images.sort()
    
    test_images = [join(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f)) and f.endswith(".jpg")]
    ground_truth_test_images = [join(ground_truth_test, f) for f in listdir(ground_truth_test) if isfile(join(ground_truth_test, f)) and f.endswith(".mat")]
    test_images.sort()
    ground_truth_test_images.sort()
    
    return train_images, ground_truth_train_images, val_images, ground_truth_val_images, test_images, ground_truth_test_images
'''
-Make train and test datasets.
-Make csv file with labels
-In 'images' key: 'id' key has the id, 'file_name' key: has the img name
-In 'categories': 'id': has the id of the class, 'name': has the class name
-In 'annotations': 'id': has the id of the img, 'bbox': the traffic sign box, 'category_id': the class id, 'ignore': if the img must be ignored
'''
import os
import json
import csv
import pandas as pd
import cv2 as cv
from tqdm import tqdm

homedir = os.path.dirname(os.path.abspath(__file__))
annotations_dir = os.path.join(homedir, 'DFG-tsd-aug-annot-json')
dataset_dir = os.path.join(homedir, 'DFG-tsd-aug-dataset')
data_train_dir = os.path.join(homedir, 'DFG-tsr-aug-train-dataset')
data_train_difficult_dir = os.path.join(
    homedir, 'DFG-tsr-aug-train-difficult-dataset')
data_test_dir = os.path.join(homedir, 'DFG-tsr-aug-test-dataset')
train_labels_dir = os.path.join(homedir, 'DFG-tsr-aug-train-labels')
test_labels_dir = os.path.join(homedir, 'DFG-tsr-aug-test-labels')
train_annotations = os.path.join(annotations_dir, 'train.json')
train_difficult_annotations = os.path.join(
    annotations_dir, 'train-difficult.json')
test_annotations = os.path.join(annotations_dir, 'test.json')


def create_dataset_from_annotations(annotations: dict, target_dir: str, src_dir=dataset_dir) -> list:
    failed_instances = []
    failed_annots = []
    ignored_annots = []
    src_images = set()

    for annot in tqdm(annotations['annotations'], desc='Creating Dataset...'):
        try:
            src_img = f'{annot["image_id"] + 1}.jpg'.zfill(11)
            src_images.add(src_img)

            if annot['ignore'] == 'true':
                ignored_annots.append(annot)
                print(f'Instance {new_img_path} ignored')
                continue
            # chdir to make cv.imread work
            new_img_path = f'{annot["id"] + 1}.jpg'.zfill(11)
            os.chdir(target_dir)
            if os.path.exists(new_img_path):
                # print(f'Image {new_img_path} already exists.')
                continue

            start_x, start_y = annot['bbox'][0], annot['bbox'][1]
            width, height = annot['bbox'][2], annot['bbox'][3]
            os.chdir(src_dir)
            image = cv.imread(src_img)
            # cv.imshow('image', image)
            # cv.waitKey(0)
            image = image[start_y:start_y + height, start_x:start_x + width, :]
            # cv.imshow('image', image)
            # cv.waitKey(0)
            image = cv.resize(image, (48, 48), interpolation=cv.INTER_AREA)
            # cv.imshow('image', image)
            # cv.waitKey(0)
            # print(f'{annot["id"] + 1}.jpg')
            os.chdir(target_dir)
            cv.imwrite(new_img_path, image)
        except:
            failed_instances.append(image)
            failed_annots.append(annot)
            # print(f'Error for: {new_img_path}')
            # print(r"cv2.error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4052: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'")
            # print('Continuing')
            continue
    return src_images, failed_instances, failed_annots, ignored_annots


def create_labels_from_annotations(annotations: dict, failed_annots: list[dict]) -> list[list]:
    labels = []
    ignored_annots = []
    for annot in tqdm(annotations['annotations'], desc='Creating labels...'):
        img_name = f'{annot["id"] + 1}.jpg'.zfill(11)
        if annot in failed_annots:
            continue
        if annot['ignore'] == 'true':
            ignored_annots.append(annot)
            print(f'Label for instance {img_name} ignored')
            continue
        # labels.append({'Filename': f'{annot["id"] + 1}.jpg'.zfill(11), 'ClassId': annot['category_id']})
        # A row of Filename, ClassId:
        labels.append(
            [img_name, annot['category_id']])
    return labels, ignored_annots


def remove_failed_annotations_from_labels(labels, failed_annots):
    df = pd.read_csv(labels)
    for annot in failed_annots:
        df.drop(df.index[annot["id"]], inplace=True)
    clean_labels_csv = df.to_csv(index=False)
    return clean_labels_csv


def save_labels_to_csv(filepath: str, labels: list[list]):
    fields = ['Filename', 'ClassId']
    with open(filepath, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(labels)


with open(train_annotations, 'r') as f:
    annotations_train = json.load(f)
with open(train_difficult_annotations, 'r') as f:
    annotations_train_difficult = json.load(f)
with open(test_annotations, 'r') as f:
    annotations_test = json.load(f)

# Make Directories if first run
if not os.path.exists(data_train_dir):
    os.mkdir(data_train_dir)
if not os.path.exists(data_train_difficult_dir):
    os.mkdir(data_train_difficult_dir)
if not os.path.exists(data_test_dir):
    os.mkdir(data_test_dir)
if not os.path.exists(train_labels_dir):
    os.mkdir(train_labels_dir)
if not os.path.exists(test_labels_dir):
    os.mkdir(test_labels_dir)


# Creaate train dataset
train_images, failed_train_instances, failed_train_annots, ignored_train = create_dataset_from_annotations(
    annotations_train, data_train_dir)
print('Training dataset created')

# Creaate train-diffficult dataset
train_difficult_images, failed_train_difficult_instances, failed_train_difficult_annots, ignored_train_difficult = create_dataset_from_annotations(
    annotations_train_difficult, data_train_difficult_dir)
print('Training-difficult dataset created')

# create test dataset
test_images, failed_test_instances, failed_test_annots, ignored_test = create_dataset_from_annotations(
    annotations_test, data_test_dir)
print('Test dataset created')

# Create labels
train_labels, ignored_train_labels = create_labels_from_annotations(
    annotations_train, failed_train_annots)
print('Training labels created')
train_difficult_labels, ignored_train_difficult_labels = create_labels_from_annotations(
    annotations_train, failed_train_difficult_annots)
print('Training-difficult labels created')
test_labels, ignored_test_labels = create_labels_from_annotations(
    annotations_test, failed_test_annots)
print('Test labels created')

save_labels_to_csv(os.path.join(
    train_labels_dir, 'train_labels.csv'), train_labels)
save_labels_to_csv(os.path.join(
    test_labels_dir, 'test_labels.csv'), test_labels)

# 16264 train images
# 2054 test images
print(f'Train images: {len(train_images)}')
print(f'Train-difficult images: {len(train_difficult_images)}')
print(f'Test images: {len(test_images)}')
print(f'DFG-tsd-aug-dataset: {len(train_images) + len(test_images)}')
print(f'Failed train instances: {len(failed_train_annots)}')
print(
    f'Failed train-difficult instances: {len(failed_train_difficult_annots)}')
print(f'Failed test instances: {len(failed_test_annots)}')
print(f'Ignored train instances: {len(ignored_train)}')
print(f'Ignored train-difficult instances: {len(ignored_train_difficult)}')
print(f'Ignored test instances: {len(ignored_test)}')
print(f'Ignored train labels: {len(ignored_train_labels)}')
print(f'Ignored train-difficult labels: {len(ignored_train_difficult_labels)}')
print(f'Ignored test labels: {len(ignored_test_labels)}')
print(f'DFG-tsr-aug-train-dataset instances: {len(train_labels)}')
print(
    f'DFG-tsr-aug-train-difficult-dataset instances: {len(train_difficult_labels)}')
print(f'DFG-tsr-aug-test-dataset instances: {len(test_labels)}')
print(
    f'All instances except failed and ignored instances: {len(train_labels) + len(test_labels)}')
print(
    f'All instances (with failed and ignored): {len(train_labels) + len(test_labels) + len(failed_test_annots) + len(failed_train_annots) + len(ignored_train) + len(ignored_train_difficult) + len(ignored_test)}')

lines = [f'Test images: {len(test_images)}',
         f'Train images: {len(train_images)}',
         f'Train-difficult images: {len(train_difficult_images)}',
         f'DFG-tsd-aug-dataset: {len(train_images) + len(test_images)}',
         f'Failed train instances: {len(failed_train_annots)}',
         f'Failed test instances: {len(failed_test_annots)}',
         f'Ignored train instances: {len(ignored_train)}',
         f'Ignored test instances: {len(ignored_test)}',
         f'Ignored train labels: {len(ignored_train_labels)}',
         f'Ignored test labels: {len(ignored_test_labels)}',
         f'DFG-tsr-aug-train-dataset instances: {len(train_labels)}',
         f'DFG-tsr-aug-test-dataset instances: {len(test_labels)}',
         f'All instances except failed and ignored instances: {len(train_labels) + len(test_labels)}',
         f'All instances (with failed and ignored): {len(train_labels) + len(test_labels) + len(failed_test_annots) + len(failed_train_annots) + len(ignored_train) + len(ignored_train_difficult) + len(ignored_test)}']

with open(os.path.join(homedir, 'DFG-size-info.txt'), 'w') as f:
    f.write('\n'.join(lines))

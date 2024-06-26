moimport tensorflow as tf

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image,mask , path, example):
    feature = {
        "image": image_feature(image),
        "mask": image_feature(mask),
        "path": bytes_feature(path),
        "area": float_feature(example["area"]),
        "bbox": float_feature_list(example["bbox"]),
        "category_id": int64_feature(example["category_id"]),
        "id": int64_feature(example["id"]),
        "image_id": int64_feature(example["image_id"]),

    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"],)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example

def generate_image( indpath ) :
    pass
def create_tfrecors(org_path ) :
    import glob
    import os
    import numpy as np
    path_lists = glob.glob(org_path + '/*.jpeg')
    for iindex , indpath in enumerate(path_lists):
        addtional_values = {
            "area": '',
            "bbox": '',
            "category_id": '',
            "id": '',
            "image_id": ''
        }
        image, mask, mask_bool  = generate_image(indpath)
        tensor_image = tf.convert_to_tensor( image , dtype=tf.int8)
        if mask_bool :
            tensor_mask = tf.convert_to_tensor(image, dtype=tf.int8)
        else :
            tensor_mask = tf.convert_to_tensor(np.arry([0]) , dtype=tf.int8)
            addtional_values['category_id'] = mask


        with tf.io.TFRecordWriter(os.path.join(
                os.path.split(org_path)[0] ,
                'tf_records',
                os.path.split(org_path)[1].replace('.jpeg','_.tfrec') )) as writer:


            example = create_example(tensor_image,tensor_mask, indpath,addtional_values)
            writer.write(example.SerializeToString())


import numpy as np

def calculate_iou(boxes1, boxes2):
    # Calculate coordinates of intersection rectangle
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    # Calculate area of intersection rectangle
    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # Calculate areas of the bounding boxes
    box1_area = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    box2_area = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

def calculate_iou_between_lists(pred_boxes, gt_boxes):
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    iou_scores = calculate_iou(pred_boxes, gt_boxes)
    return iou_scores

https://drive.google.com/file/d/1ypJ6k3z4dDe0BrN-gxSzIQ1z2baIF1zZ/view?usp=drivesdk


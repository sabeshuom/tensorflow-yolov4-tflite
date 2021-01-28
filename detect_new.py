import tensorflow as tf
import os
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes, YOLO, decode
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import im_utils
from core.config import cfg

flags.DEFINE_integer('input_size', 768, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.0, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')

flags.DEFINE_string('image', '/media/storage2/YOLO/WINE_AUS/annotations/images/wineaus_00000.png', 'path to input image')
flags.DEFINE_string('input_model_path', '/media/storage2/YOLO/WINE_AUS/models/hdf5_models/{}.h5'.format(cfg.MODEL_NAME), 'putmodel path')
flags.DEFINE_string('logdir', '/media/storage2/YOLO/WINE_AUS/logs/{}'.format(cfg.MODEL_NAME), 'log directory path')
flags.DEFINE_string('output', 'result_wa.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

flags.DEFINE_string('output_model_path', '/media/storage2/YOLO/WINE_AUS/models/hdf5_models/{}_all.h5'.format(cfg.MODEL_NAME), 'outputmodel path')



def save_tf():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    bbox_tensors = []
    prob_tensors = []
    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            elif i == 1:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
  
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    if FLAGS.framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)
    
    model = tf.keras.Model(input_layer, pred)
    model.load_weights(FLAGS.input_model_path)
    model.summary()
    # model.save(FLAGS.output_model_path)
    return model


def preprocess_image(im):
    original_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    original_image = np.array(im_utils.scale_pad_to_square(Image.fromarray(original_image), FLAGS.input_size))
    image_data = original_image / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    return images_data

def validate_train_images(_argv):
    annot_path = "/tmvcore/tensorflow-yolov4-tflite/data/dataset/wine_aus.txt"
    with open(annot_path, "r") as f:
        txt = f.readlines()
        annotations = []
        for line_txt in txt:
            line = line_txt.split()
            image_path = line[0]
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " % image_path)
            im = cv2.imread(image_path)
            boxes = np.array([list(map(int, box.split(","))) for box in line[1:]])
            bboxes = np.expand_dims(boxes[:,:4], 0).astype(np.int32)
            classes = np.expand_dims(boxes[:,4], 0)
            valid_detections = np.array([len(boxes)], dtype=np.int32)
            scores = np.array([[1.0]*len(boxes)], dtype=np.int32)
            bboxes_stack = [bboxes, scores, classes, valid_detections]
            annotated_im = utils.draw_bbox(im, bboxes_stack, is_cordinates_relative=False)
            cv2.imshow("test_image", annotated_im)
            cv2.waitKey(0)



def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    infer = save_tf()

    input_size = FLAGS.input_size
    image_path = FLAGS.image
    image_dir = '/media/storage2/YOLO/WINE_AUS/annotations/images'
    for fname in os.listdir(image_dir):
        image_path = os.path.join(image_dir, fname)
        original_image = cv2.imread(image_path)
        images_data = preprocess_image(original_image)
        batch_data = tf.constant(images_data)

        pred_bbox = infer(batch_data)
        # post processing and filtering
        # for key, value in pred_bbox.items():
        boxes = pred_bbox[:, :, 0:4]
        pred_conf = pred_bbox[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(original_image, pred_bbox)
        cv2.imshow("test image", image)
        cv2.waitKey(0)

if __name__ == '__main__':
    try:
        # app.run(main)
        app.run(validate_train_images)
    except SystemExit:
        pass

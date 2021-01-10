Skip
to
content
Search or jump
to…

Pull
requests
Issues
Marketplace
Explore


@alexivaner


farhantandia
/
Yolov4 - deepsort
2
00
Code
Issues
Pull
requests
Actions
Projects
Wiki
Security
Insights
Yolov4 - deepsort / object_tracker_optimized.py /


@farhantandia


farhantandia
enhanced
visualization and reid
Latest
commit
2097
eed
on
Dec
5, 2020
History
2
contributors


@farhantandia @ alishsuper


412
lines(349
sloc)  15.9
KB

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import filter_boxes
import core.utils as utils

# from core.functions import count_objects
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.functions import count_objects
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import pickle

import os
import cv2
import sys
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import threading
from imutils.video import VideoStream

# import for swimming style recognition
from data import DataSet
from extractor import Extractor
from tensorflow.keras.models import load_model

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils
from sklearn.svm import SVC
import tensorflow_addons as tfa

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/custom-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/identification.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.3, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('count', False, 'count objects within video')

'''---------------------------------------------------------------------------'''
class_limit = 3
sequence_length = 30

'''Style-recognition Model'''
LSTM_model = "best_model_convlstm.h5"
saved_LSTM_model = load_model(LSTM_model)

file_name = "identification"
try:
    vid = cv2.VideoCapture(int("./data/video/" + file_name + ".mp4"))
except:
    vid = cv2.VideoCapture("./data/video/" + file_name + ".mp4")

height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
vid.release()
extract_model = Extractor(image_shape=(height, width, 3))
LSTM_dataset = DataSet(seq_length=sequence_length, class_limit=class_limit, image_shape=(height, width, 3))

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# people frame buffer for LSTM
locks = {}
f = None
swimmer_buff = dict()
swimmer_pos = dict()
swimmer_style = dict()
swimmer_id = dict()
name = dict()


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def crop_and_recognize(tt, frame, map_reid):
    for track in tt:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        ID = map_reid[track.track_id]

        if locks.get(ID) == None:
            locks[ID] = threading.Lock()

        cropped0 = frame[int(bbox[1]):int(bbox[1]) + 280, int(bbox[0]):int(bbox[0]) + 180]
        cropped = cv2_clipped_zoom(cropped0, 1.5)
        cropped = cv2.resize(cropped, (100, 64))  # this is (width,height)
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("cropped.jpg", cropped)
        if cropped0.size != 0:
            locks[ID].acquire()
            if swimmer_buff.get(ID) is None:
                swimmer_buff[ID] = [cropped]
                locks[ID].release()
            else:
                swimmer_buff[ID].append(cropped)
                if len(swimmer_buff[ID]) >= sequence_length:
                    prediction_result = saved_LSTM_model.predict(np.expand_dims(swimmer_buff[ID], axis=0))
                    swimmer_style[ID] = LSTM_dataset.print_class_from_prediction(np.squeeze(prediction_result, axis=0))
                    del swimmer_buff[ID]

                locks[ID].release()
    sys.exit()


def reid(tracker, data, frame, model, le):
    map_reid = {}
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        # TODO
        cropped0 = frame[int(bbox[1]):int(bbox[1]) + 280, int(bbox[0]):int(bbox[0]) + 180]
        cropped_image = cv2_clipped_zoom(cropped0, 1.5)
        cropped_image = cv2.resize(cropped_image, (80, 180))

        # Re-ID part
        img_data = image.img_to_array(cropped_image)
        img_data = np.expand_dims(img_data, axis=0)
        img_embeddings = model.predict(img_data)

        preds = data.predict_proba(img_embeddings)[0]
        j = np.argmax(preds)
        map_reid[track.track_id] = le.classes_[j]
    return map_reid


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # load configuration for object detector
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc("XVID")
        out = cv2.VideoWriter("demo.mp4", codec, fps, (width, height))
        frame_index = -1

    # while video is running
    # swimming re-id
    model = load_model('best_model_triplet.hdf5', compile=False)
    le = pickle.loads(open('le.pickle', "rb").read())
    data = pickle.loads(open('recognizer.pikle', "rb").read())
    img_count = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        flag_ok = 0
        # map_reid = {}
        if img_count < 30 or img_count % 30 == 0:
            map_reid = reid(tracker, data, frame, model, le)
        img_count += 1
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            try:
                # ID_fix = str(map_reid[track.track_id])
                # color = colors[int(ID_fix) % len(colors)]
                # color = [i * 255 for i in color]

                if map_reid[track.track_id] == 0:
                    color = (0, 0, 0)
                elif map_reid[track.track_id] == 1:
                    color = (255, 0, 0)
                elif map_reid[track.track_id] == 2:
                    color = (255, 240, 0)
                elif map_reid[track.track_id] == 3:
                    color = (0, 255, 0)
                elif map_reid[track.track_id] == 4:
                    color = (0, 255, 255)
                elif map_reid[track.track_id] == 5:
                    color = (0, 0, 255)
                elif map_reid[track.track_id] == 6:
                    color = (127, 0, 255)
                elif map_reid[track.track_id] == 7:
                    color = (255, 182, 190)
                else:
                    color = (0, 0, 0)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[2]), int(bbox[3] + 50)), color, 2)
                cv2.putText(frame, class_name + " ReID-" + str(map_reid[track.track_id]),
                            (int(bbox[0]), int(bbox[1] - 55)), 0, 0.7, color, 2)
                flag_ok = 1
            except Exception as e:
                # print("Exception", map_reid, track.track_id)
                # print("Exception", e)
                map_reid = reid(tracker, data, frame, model, le)

                # update tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    cv2.putText(frame, class_name + " ReID-" + str(map_reid[track.track_id]),
                                (int(bbox[0]), int(bbox[1] - 55)), 0, 0.7, color, 2)
                print("after", map_reid, track.track_id)
                pass

        # --------Threading------------#
        if flag_ok == 1:
            thread_num = 32
            threads = [None] * thread_num
            original_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            f = original_frame
            n_t = len(tracker.tracks)
            s = 0
            e = n_t // thread_num
            for i in range(thread_num):
                if i == thread_num - 1:
                    threads[i] = threading.Thread(target=crop_and_recognize,
                                                  args=(tracker.tracks[:-1], original_frame, map_reid,))
                else:
                    threads[i] = threading.Thread(target=crop_and_recognize,
                                                  args=(tracker.tracks[s:e], original_frame, map_reid,))
                s = e
                e += e

            for t in threads:
                t.start()

            for i in range(len(tracker.tracks)):
                track = tracker.tracks[i]
                bbox = track.to_tlbr()
                try:
                    ID = map_reid[track.track_id]
                except Exception as e:
                    # print("map_reid, track.track_id", map_reid, track.track_id)
                    # print("Exception 2", e)
                    pass

                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                    # color = colors[ID % len(colors)]
                # color = [i * 255 for i in color]
                # cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                if swimmer_style.get(ID) is not None:
                    (sstyle, conf) = swimmer_style[ID]
                    cv2.putText(f, str(sstyle) + ": " + str(int(round(conf, 2) * 100)) + "%",
                                (int(bbox[0]), int(bbox[1] - 35)), 0, 0.7, (255, 255, 255), 2)
            frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        else:
            print("NOT OKAY", "map_reid, track.track_id", map_reid, track.track_id, "len(tracker.tracks)",
                  len(tracker.tracks))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
            frame_index = frame_index + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cv2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from utils.model_utils import perform_prediction, decode_predictions

parts = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle"
]

min_conf_score = 0.2
model_path = 'models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'

# Resolution = ((InputImageSize - 1) / OutputStride) + 1
# (513 - 1 / 32) + 1 = 17 (our case), so we are using the "worst", accuracy wise

interpreter = tf.lite.Interpreter(model_path=model_path)

image = cv2.imread('images/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

heatmaps, offsets, displacements_fwd, displacements_bwd, resized_image = perform_prediction(image, interpreter)
keypoints = decode_predictions(heatmaps, offsets, output_stride=32)

resize_y_ratio = image.shape[0]/resized_image.shape[0]
resize_x_ratio = image.shape[1]/resized_image.shape[1]

image_cpy = np.copy(image)

pose_conf = np.array([keypoint["confidence"] for keypoint in keypoints]).mean()

for keypoint in keypoints:
    scale = 5

    # rescale to original (not resized by model) image coordinates
    pos_y = int(keypoint["y"] * resize_y_ratio)
    pos_x = int(keypoint["x"] * resize_x_ratio)
    confidence_score = keypoint["confidence"]
    if confidence_score > min_conf_score:
        cv2.circle(image_cpy, (pos_x, pos_y), scale, (255, 0, 0), thickness=cv2.FILLED)
        cv2.putText(image_cpy, parts[keypoint["part_index"]], (pos_x, pos_y), 0, 0.4, (0, 255, 0))

    print("Confidence for {}: {}".format(parts[keypoint["part_index"]], confidence_score))

print("Confidence for pose {}".format(pose_conf))

plt.imshow(image_cpy)
plt.show()

import cv2
import numpy as np
from utils.math_utils import sigmoid, argmax2d

def perform_prediction(image_in, model):
    """
    Performs prediction given an input image and tf.lite.Interpreter instance.
    Inputs:
        image_in: input image (in 0..255 intensity range)
        model: tf.lite.Interpreter instance
    Outputs:
        heatmaps: tensor containing heatmaps for each body part
        offsets: tensor containing offsets for each body part
        displacement_fwd: tensor containing forward displacements
        displacement_bwd: tensor containing backward displacements
        image: copy of the image (resized) that was fed to the model

    """
    # work on copy
    image = np.copy(image_in)
    # allocate tensors
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model_req_size = np.array((input_details[0]['shape'][2], input_details[0]['shape'][1]))

    image = cv2.resize(image, tuple(model_req_size))
    image = (image - 127.5) / 127.5

    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    model.set_tensor(input_details[0]['index'], input_data)
    # predict
    model.invoke()
    # retrieve outputs
    heatmaps = model.get_tensor(output_details[0]['index'])
    # res x res x 34 values, first 17 are x offsets, remaining 17 are y offsets (same scale as image)
    offsets = model.get_tensor(output_details[1]['index'])
    displacement_fwd = model.get_tensor(output_details[2]['index'])
    displacement_bwd = model.get_tensor(output_details[3]['index'])

    # return heatmaps, offsets, displacements and the resized image
    return heatmaps, offsets, displacement_fwd, displacement_bwd, image

def decode_predictions(heatmaps, offsets, output_stride=32):
    """
    Decodes model's predictions, returns a list of keypoints, one per body part.
    Inputs:
        heatmaps: predicted heatmaps
        offsets: predicted offsets
        output_stride: parameter depending on model (default: 32)
    Outputs:
        keypoints: list of keypoints, each keypoint is a dict in the following format
                   {
                       "x": x component of keypoint, 
                       "y": y component of keypoint, 
                       "part_index": body part index as defined in PoseNet specification, 
                       "confidence": confidence score
                   }
    """
    # squeeze in case of single image tensor
    heatmaps = np.squeeze(heatmaps)
    offsets = np.squeeze(offsets)

    # score map
    scores = sigmoid(heatmaps)
    # hmap positions is an 17x2 array obtained by argmax2d the scores vector
    heatmap_positions = argmax2d(scores)
    keypoints = []
    # for each body part, we know its position on the heatmap
    for part_index, hmap_pos in enumerate(heatmap_positions):
        # we need to find offsets
        offset_y = offsets[hmap_pos[0]][hmap_pos[1]][part_index]
        offset_x = offsets[hmap_pos[0]][hmap_pos[1]][part_index + 17]
        # use the offsets to compute the keypoints
        kp_y = int(hmap_pos[0] * output_stride + offset_y)
        kp_x = int(hmap_pos[1] * output_stride + offset_x)
        conf_score = scores[hmap_pos[0]][hmap_pos[1]][part_index]
        
        keypoints.append({"x": kp_x, "y": kp_y, "part_index": part_index, "confidence": conf_score})
    return keypoints

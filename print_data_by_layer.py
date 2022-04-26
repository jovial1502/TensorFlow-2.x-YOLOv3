import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, image_preprocess
from yolov3.configs import *
import sys

OUTPUT_FOLDER = "layer_outputs/"
PRINT_SUMMARY = False # print network summary to console


def main():
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
    print("YOLO is loaded")

    if PRINT_SUMMARY:
        print("===================\n")
        print("Network summary: \n")
        print(yolo.summary())
        print("===================\n")

    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    print("Testset is loaded")

    for index in range(1):
        print(f"Working with index: {index}")
        ann_dataset = testset.annotations[index]

        image_name = ann_dataset[0].split('/')[-1]
        original_image, bbox_data_gt = testset.parse_annotation(ann_dataset, True)
        print(f"Image: {image_name}")
        
        image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)

        print("Starting prediction...")

        # set treshold to max to avoid truncation of arrays when printing to a file
        np.set_printoptions(threshold=sys.maxsize)

        for i in range(1, 44):
            print(f"Getting output for layer {i}")
            tmp_model = tf.keras.Model(yolo.layers[0].input, yolo.layers[i].output)
            tmp_output = tmp_model.predict(image_data)[0]

            # construct filename for every layer, e.g. 9 conv2d_2.txt, where 9 is layer number and conv2d_2 is layer name
            filename = str(i) + " " + yolo.get_layer(index=i).name + ".txt"
            filepath = OUTPUT_FOLDER + filename 

            f = open(filepath, "a")
            f.write(str(tmp_output))
            f.close()

        # restore treshold
        np.set_printoptions(threshold=False)

if __name__ == "__main__":
    main()

    


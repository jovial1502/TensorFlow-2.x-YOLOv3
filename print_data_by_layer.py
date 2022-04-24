from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, image_preprocess, postprocess_boxes, nms, read_class_names
from yolov3.configs import *
import shutil
import json
import time
import gc
import sys



def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

if __name__ == "__main__":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
    print("YOLO is loaded")
    print(yolo.summary())

    # for i in range(1, len(yolo.layers)):
    #     print(yolo.get_layer(index=i).name)



    # testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    # print("Testset is loaded")

    # for index in range(1):
    #     print(f"Working with index: {index}")
    #     ann_dataset = testset.annotations[index]

    #     image_name = ann_dataset[0].split('/')[-1]
    #     original_image, bbox_data_gt = testset.parse_annotation(ann_dataset, True)
    #     print(f"Image: {image_name}")
        
    #     image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
    #     image_data = image[np.newaxis, ...].astype(np.float32)

    #     print("Starting prediction...")

    #     np.set_printoptions(threshold=sys.maxsize)

    #     layer_outputs = {}
    #     for i in range(1, 44):
    #         print(f"Getting output for layer {i}")
    #         tmp_model = tf.keras.Model(yolo.layers[0].input, yolo.layers[i].output)
    #         tmp_output = tmp_model.predict(image_data)[0]
    #         #tmp_size = total_size(tmp_output)
    #         #layer_outputs[yolo.get_layer(index=i).name] = tmp_output

    #         filename = "layer_outputs/" + str(i) + " " + yolo.get_layer(index=i).name + ".txt"
    #         f = open(filename, "a")
    #         f.write(str(tmp_output))
    #         f.close()

    #     np.set_printoptions(threshold=False)




        
        # f = open("layer_outputs.txt", "a")
        # f.write(str(layer_outputs))
        # f.close()

        # # pred_bbox = yolo.predict(image_data)
        # print("Prediction ended")


    


from yolo_core.config import cfg
import numpy as np
import time


def get_anchors(anchors, tiny=False):
    # get anchors as a numpy array
    anchors = np.array(anchors)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def load_config(labels_path, is_tiny):
    """ Loads yolo config parameters"""
    
    print("[YOLO-UTILS] Loading yolo config parameters...")
    config_start = time.time()

    if is_tiny:
        # get strides anchos and xyscale from config file
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, is_tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS, is_tiny)
        XYSCALE = cfg.YOLO.XYSCALE
    
    # get the number of classes from 
    NUM_CLASS = len(read_class_names(labels_path))
    print("[YOLO-UTILS] Yolo parameters loaded succesfully, it took {} ms"
            .format((time.time()-config_start)*1000))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def read_class_names(class_file_name):
    """ Read the class name as dictionary that contains the index
        as key and the class name as value"""
    
    print("[YOLO-UTILS] Reading class names...")
    read_start = time.time()
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    
    print(  "[YOLO-UTILS] Class read succesfully, it took {} ms"
            .format((time.time()-read_start)*1000))
    return names
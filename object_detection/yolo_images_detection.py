import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.saved_model import tag_constants
import argparse
import logging
from yolo_inferencev1 import YoloInferator
#from yolo_inferencev2 import YoloInferator

if __name__ == '__main__':
    """ !Important!
        Before executing the script, run the following command from the shell

        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

        or add it to .bashrc file. If not an error will be prompted from cv2"""

    # Register a tf logger
    logging.getLogger("tensorflow").setLevel(logging.DEBUG)    
    parser = argparse.ArgumentParser(description="Perform Inference on a TF model")

    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing the yolo saved model in .pb format')

    parser.add_argument('--images_path', type=str, default='dogs.jpeg',
                        help='Path to the images to perform detection')

    parser.add_argument('--gpu_mem', type=int, default=0,
                        help='Upper bound for GPU memory in MB'
                        '0 means that will allow GPU memory to growth')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold to draw bounding boxes on image')

    parser.add_argument('--label', type=str, default='../models/coco_labels.txt',
                        help='Path to labels of the model')

    parser.add_argument('--iou', type=float, default=0.45,
                        help='Iou threshold')

    parser.add_argument('--tiny', action='store_true',
                        help='Specify if use the tiny yolo model')

    parser.add_argument('--opencv', action='store_true',
                        help='Specify if use opencv backend to load image')

    args = parser.parse_args()

    print("[MAIN] Starting detection pipeline...")
    start_time=time.time()

    # Instantiate the inferator class 
    inferator = YoloInferator()

    # config the gpu memory usage limits
    inferator.config_gpu_memory(args.gpu_mem)

    # load model details
    inferator.load_model_details(args.label, args.tiny, args.model_dir)

    # load a graph func from a saved model to perform inference
    #inferator.get_func_from_saved_model(args.model_dir)

    # perform inference on given image
    inferator.run_inference_on_images(model_dir=args.model_dir, images_path=args.images_path, 
                                        threshold=args.threshold, iou=args.iou, warmup_iters=10,
                                        opencv=args.opencv)     



    print("[MAIN] Detection pipeline finished."
        + "Total time elapsed: {} seconds"
        .format(time.time()-start_time))
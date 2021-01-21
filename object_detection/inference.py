""" !Important!
    Before executing the script, run the following command from the shell

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

    or add it to .bashrc file. If not an error will be prompted from cv2"""

import cv2
import tensorflow as tf
import numpy as np
import glob, os, sys
import re
import pandas as pd
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

import time
import image_utils as imgutils
from results_utils import BboxResults, PerformanceResults, init_common_metadata, save_common_results

class Inferator():
    def __init__(self, verbose=1):
        self.attributes = {}

    def get_func_from_saved_model(self, saved_model_dir):
        """ Gets the graph function to perform inference
            from a saved .pb model """
        
        if "FP16" in saved_model_dir or "fp16" in saved_model_dir:
            print("[MODEL] Model precision recognized as FP16")
            self.attributes["precision"] = "FP16"
        elif "INT8" in saved_model_dir or "in8" in saved_model_dir:
            print("[MODEL] Model precision recognized as INT8")
            self.attributes["precision"] = "INT8"
        elif "FP32" in saved_model_dir or "fp32" in saved_model_dir:
            print("[MODEL] Model precision recognized as FP32")
            self.attributes["precision"] = "FP32"
        else:
            print("[MODEL] Model precision unknown")
            self.attributes["precision"] = "UNKNOWN"

        print("[MODEL] Loading saved model from {}...".format(saved_model_dir))
        model_start = time.time()
        # Load the saved model
        saved_model_loaded = tf.saved_model.load(
            saved_model_dir, tags=[tag_constants.SERVING])

        print("[MODEL] Succesfully loaded saved model from {}...".format(saved_model_dir))
        print("[MODEL] Model loaded in {} ms".format((time.time()-model_start)*1000))

        print("[MODEL] Creating graph function from model...")
        graph_func_time = time.time()
        # Creates a graph function from the saved model
        #graph_func = saved_model_loaded.signatures[
        #    list(saved_model_loaded.signatures.keys())[0]]

        print("[MODEL] Succesfully created graph func from model")
        print("[MODEL] Graph func loading took {} ms"
                .format((time.time()-graph_func_time)*1000))

        #self.attributes["graph_func"] = graph_func
        self.attributes["saved_model_loaded"] = saved_model_loaded
        return saved_model_loaded #graph_func

    def config_gpu_memory(self, gpu_mem_cap):
        """ Function to set the memory usage of the gpu to perform inference"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            return
        print('[GPU-CONFIG] Found the following GPUs:')
        for gpu in gpus:
            print('[GPU-CONFIG]  ', gpu)
        for gpu in gpus:
            try:
                if not gpu_mem_cap:
                    # Allow for memory to growth as requires
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("[GPU-CONFIG] Configured to allow memory growth")
                else:
                    # Explicitly states a memory limit
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_mem_cap)])
                    print("[GPU-CONFIG] Configured max memory limit of to {}"
                            .format(gpu_mem_cap))
            except RuntimeError as e:
                print('[GPU-CONFIG] Can not set GPU memory config', e)
    
    def load_model_details(self, model_dir):
        details_path = model_dir + 'details.txt'
        with open(details_path, 'r') as f:
            for line in f.readlines():
                splitted = line.split('=')
                key = splitted[0]
                value = splitted[1]

                if value[-1] == "\n":
                    #print("[MODEL] Repairing key {} with value {}".format(key, value))
                    value = value[0:-1]

                print("[MODEL] Adding key {} with value {} to attributes"
                    .format(key, value))
                self.attributes[key] = value
                #print("[MODEL] Key loaded? {}".format(self.attributes[key]))


    def warmup(self, warmup_iters=10, opencv=False):
        #print("[INFERENCE] Starting warmup on {} iterations..."
        #        .format(warmup_iters))
        #warmup_start = time.time()
        # get input size from model attributes
        #input_size = int(self.attributes["INPUT_SIZE"])

        self.run_inference(labels = ["pig", "person"], warmup_iters=10, opencv=opencv)
        
    def run_inference(self, image_path=None, labels=None, bbox_results=None, performance_results=None, warmup_iters=0, threshold=0.5, image=None, resize=False, opencv=False):   
        input_size = int(self.attributes["INPUT_SIZE"])
        # Load image with opencv backend
        if image_path is not None:
            if opencv:
                print("[INFERENCE] Loading image with opencv backend...")
                # opencv option
                image, total_image_loading = imgutils.load_image_as_np(image_path, "CV2")

                # preprocess image to work on tf
                print("[INFERENCE] Preprocessing image...")
                start_preprocessing = time.time()

                # resize image to netwrk input dimensions
                if resize:
                    resized_image = imgutils.resize_image(image, (input_size, input_size))
                else:
                    resized_image = image

                # conventional conversion (use with opencv option)
                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(resized_image)
                #input_tensor = tf.convert_to_tensor(image)
                # The model expects a batch of images, so add an axis with `tf.newaxis`.
                input_tensor = input_tensor[tf.newaxis, ...]

                total_preprocessing = (time.time()-start_preprocessing)*1000
                print("[INFERENCE] Preprocessing done!, it took {}ms"
                    .format(total_preprocessing))


            # tf backend
            else: 
                print("[INFERENCE] Loading image with tf backend...")
                if resize:
                    # dataset option
                    dataset, total_image_loading, total_preprocessing = imgutils.get_dataset_tf(image_path=image_path, 
                                                                                            input_size=input_size)
    
                else:
                    # dataset option
                    dataset, total_image_loading, total_preprocessing = imgutils.get_dataset_tf(image_path=image_path)
                # take the batched image
                dataset_enum = enumerate(dataset)
                input_tensor = list(dataset_enum)[0][1]

                # take image as np and convert to rgb
                image_bgr = input_tensor.numpy()[0]
                image = image_bgr[...,::-1].copy()
      
        # get a copy of the graph func
        #graph_func = self.attributes["graph_func"]
        saved_model_loaded = self.attributes["saved_model_loaded"]

        if warmup_iters == 0:
            print("[INFERENCE] Now performing inference...")
            inference_start_time = time.time()
            
            # get the detections
            detections = saved_model_loaded(input_tensor)

            total_inference = (time.time()-inference_start_time)*1000
            print("[INFERENCE] Inference took {} ms"
                .format(total_inference))

            # TODO: ADD tracker updating 

            # draw results on image
            if (bbox_results is not None) and (performance_results is not None):
                drawing_time = imgutils.draw_bounding_boxes(image, detections, 
                                            labels, threshold, bbox_results=bbox_results)
                
                height, width, _ = image.shape
                # add new performance results to object
                performance_results.add_new_result(width, height, total_image_loading, 
                                                total_preprocessing, total_inference, drawing_time)

            else :
                # draw case of just one image
                imgutils.draw_bounding_boxes(image, detections, labels, threshold, save_results=True)
        else:            
            warmup_start = time.time()
            if opencv:
                print("[WARMUP] Starting warmup with opencv backend on {} iters"
                    .format(warmup_iters))
                
                for i in range(warmup_iters):
                    print("[WARMUP] Generating image {} with dims {}x{}"
                            .format(i+1, input_size, input_size))

                    image = np.random.normal(size=(input_size, input_size, 3)).astype(np.uint8)
                    # conventional conversion (use with opencv option)
                    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                    input_tensor = tf.convert_to_tensor(image)
                    # The model expects a batch of images, so add an axis with `tf.newaxis`.
                    input_tensor = input_tensor[tf.newaxis, ...]
                    
                    # get the detections
                    print("[WARMUP] Now performing warmup inference...")
                    inference_start_time = time.time()
                    detections = saved_model_loaded(input_tensor)
                    print("[WARMUP] Warmup inference took {} ms"
                            .format((time.time()-inference_start_time)*1000))

                    _ = imgutils.draw_bounding_boxes(image, detections, labels, 0.05)

            else:
                print("[WARMUP] Starting warmup on with tf backend on {} iterations..."
                    .format(warmup_iters))

                for i in range(warmup_iters):
                    print("[WARMUP] Generating image {} with dims {}x{}"
                            .format(i+1, input_size, input_size))
                    features = np.random.normal(loc=112, scale=70,
                            size=(1, input_size, input_size, 3)).astype(np.float32)

                    print("[WARMUP] Creating features...")
                    features = np.clip(features, 0.0, 255.0).astype(np.uint8)
                    features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
                                        "features", initializer=tf.constant(features)))
                    print("[WARMUP] Creating dataset from features...")
                    dataset = tf.data.Dataset.from_tensor_slices([features])
                    dataset = dataset.repeat(count=1)
                    dataset_enum = enumerate(dataset)

                    print("[WARMUP] Retrieving image and input tensor...")
                    # get input tensor and cast to image (np)
                    input_tensor = list(dataset_enum)[0][1]
                    image = input_tensor.numpy()[0]

                    # get the detections
                    print("[WARMUP] Now performing warmup inference...")
                    inference_start_time = time.time()
                    detections = saved_model_loaded(input_tensor)
                    print("[WARMUP] Warmup inference took {} ms"
                            .format((time.time()-inference_start_time)*1000))

                    #perform drawing warmup with very low threshold
                    _ = imgutils.draw_bounding_boxes(image, detections, labels, 0.05)
                    
                    # display results in ms
                    print("[WARMUP] Warmup finished, it took {} ms"
                            .format((time.time()-warmup_start)*1000))

    def run_inference_on_images(self, images_path, labels=None, threshold=0.3, resize=False, opencv=False):
        # define frame count and a helper function to read
        # the images sorted by numerical index
        frame_count = 1
        numbers = re.compile(r'(\d+)')
        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
        
        # read image sorted by numerical order
        image_paths = sorted(glob.glob(images_path + "*.png"), key=numericalSort)
        print("[INFERENCE] Found {} images in {} ...".format(len(image_paths), images_path))

        #create a class to store results
        bbox_results = BboxResults()
        performance_results = PerformanceResults()

        # Iterate over all images, perform inference and update 
        # results dataframe
        for image_path in image_paths:
            print("[INFERENCE] Processing frame/image {} from {}".format(frame_count, image_path))
            init_common_metadata(bbox_results, performance_results, image_path, frame_count)

            # perform inference on image and get the detection results
            self.run_inference(image_path = image_path, labels=labels, 
                                bbox_results=bbox_results, performance_results=performance_results,
                                resize=resize, opencv=opencv)

            print("[INFERENCE] Image/frame {} processed".format(frame_count))
            frame_count += 1

        print("[INFERENCE] All frames procesed!")

        output_path = "{}results/{}".format(images_path, self.attributes["MODEL_NAME"])
        if opencv:
            output_path += "/opencv-backend"
        else:
            output_path += "/tf-backend"
        
        # save results obtained from performance and detections to output
        save_common_results(bbox_results, performance_results, output_path, 
                            self.attributes["MODEL_NAME"], self.attributes["precision"], 
                            threshold, resize=resize, opencv=opencv)




    

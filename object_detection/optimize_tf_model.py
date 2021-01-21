""" !Important!
    Before executing the script, run the following command from the shell

    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

    or add it to .bashrc file. If not an error will be prompted from cv2"""

from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import argparse
import logging
import sys
import cv2
import image_utils as imgutils
import time
import os
from pathlib import Path
import pandas as pd
from shutil import copyfile

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS

IMAGES_PATH = None
INPUT_SIZE = None

CALIBRATION_TIME = None

# function to pass as argument to the builder function
# it will be used to optimize the network based on given examples
def calibration_fn():
    """function to pass as argument to the builder function
    it will be used to optimize the network based on given examples """
    print("[CALIBRATION] Starting calibration process...")
    
    images_found = sorted(os.listdir(IMAGES_PATH))
    print("[CALIBRATION] Obtaining calibration images from {}"
        .format(images_found))
    print("[CALIBRATION] Done! Found {} images for calibration"
            .format(len(images_found)))

    print("[CALIBRATION] Starting image yielding...")
    start_yielding = time.time()
    for image_path in images_found:
        input_image = imgutils.read_image_from_cv2(IMAGES_PATH + image_path)
        resized_image = imgutils.resize_image(input_image, (INPUT_SIZE, INPUT_SIZE))

        final_image = resized_image[np.newaxis, ...].astype("uint8")

        print("[CALIBRATION] Yielding image from {}".format(image_path))
        start_yielding = time.time()
        yield (final_image,)
        print("[CALIBRATION] Image yielding Done, it took {} ms"
            .format((time.time()-start_yielding)*1000))

    CALIBRATION_TIME = (time.time()-start_yielding)*1000
    print("[CALIBRATION] Calibration procces finished, it took {} ms"
        .format(CALIBRATION_TIME))

def batched_calibration_fn():
    """function to pass as argument to the builder function
    it will be used to optimize the network based on given examples """
    print("[CALIBRATION] Starting calibration process...")
    
    images_found = sorted(os.listdir(IMAGES_PATH))
    print("[CALIBRATION] Obtaining calibration images from {}"
        .format(images_found))
    print("[CALIBRATION] Done! Found {} images for calibration"
            .format(len(images_found)))

    print("[CALIBRATION] Starting image yielding...")
    start_calibration = time.time()
    batched_input = np.zeros((len(images_found), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for input_value in range(len(images_found)):

        # read and resize the image
        input_image = imgutils.read_image_from_cv2(IMAGES_PATH + images_found[input_value])
        image_data = imgutils.preprocess_image(input_image, (INPUT_SIZE, INPUT_SIZE))

        # add a new axis to match requested shape
        final_image = image_data[np.newaxis, ...].astype("uint8")

        # add image to batched images
        batched_input[input_value, :] = final_image

        print("[CALIBRATION] Adding image {} from {}".format(input_value+1, images_found[input_value]))
        start_adding = time.time()
        
        print("[CALIBRATION] Image adding step Done, it took {} ms"
            .format((time.time()-start_adding)*1000))

    print("[CALIBRATION] Yielding batched input")
    start_yielding = time.time()
    yield (final_image,)
    print("[CALIBRATION] Image yielding Done, it took {} ms"
        .format((time.time()-start_yielding)*1000))
    
    CALIBRATION_TIME = (time.time()-start_calibration)*1000
    print("[CALIBRATION] Calibration procces finished, it took {} ms"
        .format(CALIBRATION_TIME))

if __name__ == '__main__':
    # define a tf logger with high verbosity
    logging.getLogger("tensorflow").setLevel(logging.DEBUG) 

    # define the args to receive
    parser = argparse.ArgumentParser(description="Perform Inference on a TF model")

    parser.add_argument('--model_dir', type=str, default=None,
                        required=True,
                        help='Directory containing the saved model in .pb format')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Precision mode to optimize the mdel')

    parser.add_argument('--images_path', type=str, required=True,
                        help='Path to the images to perform calibration')

    parser.add_argument('--precision_mode', type=str, default="FP32",
                        help='Precision mode to optimize the mdel')
    
    parser.add_argument('--input_size', type=int, required=True,
                        help='Input size to use in the calibration step')

    args = parser.parse_args()

    allowed_precision_modes = ["FP32", "FP16", "INT8"]
    precision_mode = args.precision_mode.upper()

    precision_modes = {"FP32": trt.TrtPrecisionMode.FP32, 
                        "FP16": trt.TrtPrecisionMode.FP16,
                        "INT8": trt.TrtPrecisionMode.INT8}

    # check for sanity of precision mode provided
    if precision_mode not in allowed_precision_modes:
        print("[OPTIMIZATION] Precision mode is invalid. Please select" 
            + "one from the list: {}".format(allowed_precision_modes))
        sys.exit()
    else:
        print("[OPTIMIZATION] Precision mode selected: {}"
            .format(precision_mode))

    precision_mode = precision_modes[precision_mode]
        
    IMAGES_PATH = args.images_path
    INPUT_SIZE = args.input_size

    print("[OPTIMIZATION] Starting optimization tasks...")
    start_optimization = time.time()

    # get the name of the folder containing the model
    # and append to the output dir
    model_dir = Path(args.model_dir).parts[-1]
    print("[DIR-CREATION] Model dir: {}".format(model_dir))
    output_dir = args.output_dir
    if output_dir[len(output_dir)-1] != "/":
        output_dir += '/'

    
    output_dir += "{}_{}/".format(model_dir, precision_mode)

    # creating the output directory that will contain the model
    print("[DIR-CREATION] Attempting to create output directory...")
    print("[DIR-CREATION] Model will be saved under {}"
        .format(output_dir))
    if not os.path.exists(output_dir):
        print("[DIR-CREATION] Directory doesnt exists, creating...") 
        os.makedirs(output_dir)
        print("[DIR-CREATION] Directory created succesfully...") 
    else:
        print("[DIR-CREATION] Output directory already exists...")

    # define the optimization parameters
    print("[OPTIMIZATION] Defining optimization parameters...")
    start_params = time.time()
    conversion_params = conversion_params._replace(
                        max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    conversion_params = conversion_params._replace(maximum_cached_engines=100)
    params_time = (time.time()-start_params)*1000
    print("[OPTIMIZATION] Optimization parameters defined, it took {} ms"
        .format(params_time))

    # create the converter object
    print("[OPTIMIZATION] Creating TRT converter...")
    start_converter = time.time()
    converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=args.model_dir,
            conversion_params=conversion_params)
    converter_time = (time.time()-start_converter)*1000
    print("[OPTIMIZATION] Converter created succesfully, it took {} ms"
        .format(converter_time))

    # build the trt graph
    print("[OPTIMIZATION] Building TRT graph...")
    start_build=time.time()
    
    if precision_mode == precision_modes["INT8"]:
        converter.convert(calibration_input_fn=calibration_fn)
    else:
        converter.convert()
    converter.build(input_fn=batched_calibration_fn)
    build_time = (time.time()-start_build)*1000
    print("[OPTIMIZATION] Building TRT graph done!, it took {} ms"
        .format(build_time))
    
    # save the trt graph
    print("[OPTIMIZATION] Saving TRT optimized model...")
    start_save = time.time()
    converter.save(output_dir)
    save_time = (time.time()-start_save)*1000
    print("[OPTIMIZATION] Model saved succesfully, it took {} ms"
        .format(save_time))

    optimization_time = (time.time()-start_optimization)*1000
    print("[OPTIMIZATION] All optimization tasks completed. Total time elapsed: {} ms"
        .format(optimization_time))

    results = [model_dir, args.model_dir, params_time, 
                converter_time, CALIBRATION_TIME, build_time, 
                save_time, optimization_time]
    columns = ["Model", "Model_dir", "params_time",
                "conv_creation_time", "calibration_time",
                "build_time", "save_time", "total_optim_time"]
    results_df = pd.DataFrame([results], columns = columns)
    results_df.to_excel(output_dir + '/optimization_benchmark.xlsx', index=False)
    copyfile(args.model_dir+'details.txt', output_dir+'details.txt')


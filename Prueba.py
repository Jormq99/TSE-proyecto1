import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from openvino.runtime import Core

sys.path.append("openvino_notebooks/notebooks/utils")
import notebook_utils as utils

# A directory where the model will be downloaded.
base_model_dir = "model"
# The name of the model from Open Model Zoo.
detection_model_name = "vehicle-detection-0200"
recognition_model_name = "vehicle-attributes-recognition-barrier-0039"
# Selected precision (FP32, FP16, FP16-INT8)
precision = "FP32"

# Check if the model exists.
detection_model_path = (
    f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"
)
recognition_model_path = (
    f"model/intel/{recognition_model_name}/{precision}/{recognition_model_name}.xml"
)

# Download the detection model.
if not os.path.exists(detection_model_path):
    download_command = f"omz_downloader " \
                       f"--name {detection_model_name} " \
                       f"--precision {precision} " \
                       f"--output_dir {base_model_dir}"
    subprocess.run(download_command, shell=True)
# Download the recognition model.
if not os.path.exists(recognition_model_path):
    download_command = f"omz_downloader " \
                       f"--name {recognition_model_name} " \
                       f"--precision {precision} " \
                       f"--output_dir {base_model_dir}"
    subprocess.run(download_command, shell=True)


import ipywidgets as widgets

core = Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device


# Initialize OpenVINO Runtime runtime.
core = Core()


def model_init(model_path: str) -> Tuple:
    """
    Read the network and weights from file, load the
    model on the CPU and get input and output names of nodes

    :param: model: model architecture path *.xml
    :retuns:
            input_key: Input node network
            output_key: Output node network
            exec_net: Encoder model network
            net: Model network
    """

    # Read the network and corresponding weights from a file.
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    # Get input and output names of nodes.
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model


# de -> detection
# re -> recognition
# Detection model initialization.
input_key_de, output_keys_de, compiled_model_de = model_init(detection_model_path)
# Recognition model initialization.
input_key_re, output_keys_re, compiled_model_re = model_init(recognition_model_path)

# Get input size - Detection.
height_de, width_de = list(input_key_de.shape)[2:]
# Get input size - Recognition.
height_re, width_re = list(input_key_re.shape)[2:]

def plt_show(raw_image):
    """
    Use matplot to show image inline
    raw_image: input image

    :param: raw_image:image array
    """
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(raw_image)


    # Load an image.
url = "https://storage.openvinotoolkit.org/data/test_data/images/person-bicycle-car-detection.bmp"
filename = "cars.jpg"
directory = "data"
image_file = utils.download_file(
    url, filename=filename, directory=directory, show_progress=False, silent=True,timeout=30
)
assert Path(image_file).exists()

# Read the image.
image_de = cv2.imread("data/cars.jpg")
# Resize it to [3, 256, 256].
resized_image_de = cv2.resize(image_de, (width_de, height_de))
# Expand the batch channel to [1, 3, 256, 256].
input_image_de = np.expand_dims(resized_image_de.transpose(2, 0, 1), 0)
# Show the image.
plt_show(cv2.cvtColor(image_de, cv2.COLOR_BGR2RGB))

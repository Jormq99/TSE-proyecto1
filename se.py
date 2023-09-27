import json
import math
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Any, List, Dict
from IPython.display import Video
import openvino as ov

sys.path.append("openvino_notebooks/notebooks/utils")
from notebook_utils import download_file

DATA_DIR = Path("data/")
MODEL_DIR = Path("model/")
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_NAME = "slowfast_r50"
MODEL_REPOSITORY = "facebookresearch/pytorchvideo"
DEVICE = "cpu"

# load the pretrained model from the repository
model = torch.hub.load(
    repo_or_dir=MODEL_REPOSITORY, model=MODEL_NAME, pretrained=True, skip_validation=True
)

# set the device to allocate tensors to. for example, "cpu" or "cuda"
model.to(DEVICE)

# set the model to eval mode
model.eval()


CLASSNAMES_SOURCE = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)
CLASSNAMES_FILE = "kinetics_classnames.json"

download_file(url=CLASSNAMES_SOURCE, directory=DATA_DIR, show_progress=True)

# load from json
with open(DATA_DIR / CLASSNAMES_FILE, "r") as f:
    kinetics_classnames = json.load(f)

# load dict of id to class label mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

VIDEO_SOURCE = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
VIDEO_NAME = "archery.mp4"
VIDEO_PATH = DATA_DIR / VIDEO_NAME

download_file(url=VIDEO_SOURCE, directory=DATA_DIR, show_progress=True)
Video(VIDEO_PATH, embed=True)

def scale_short_side(size: int, frame: np.ndarray) -> np.ndarray:
    """
    Scale the short side of the frame to size and return a float
    array.
    """
    height = frame.shape[0]
    width = frame.shape[1]
    # return unchanged if short side already scaled
    if (width <= height and width == size) or (height <= width and height == size):
        return frame
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    scaled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return scaled.astype(np.float32)


def center_crop(size: int, frame: np.ndarray) -> np.ndarray:
    """
    Center crop the input frame to size.
    """
    height = frame.shape[0]
    width = frame.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = frame[y_offset:y_offset + size, x_offset:x_offset + size, :]
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


def normalize(array: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Normalize a given array by subtracting the mean and dividing the std.
    """
    if array.dtype == np.uint8:
        array = array.astype(np.float32)
        array = array / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    array = array - mean
    array = array / std
    return array


def pack_pathway_output(frames: np.ndarray, alpha: int = 4) -> List[np.ndarray]:
    """
    Prepare output as a list of arrays, each corresponding
    to a unique pathway.
    """
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = np.take(
        frames,
        indices=np.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).astype(np.int_),
        axis=1
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list


def process_inputs(
    frames: List[np.ndarray],
    num_frames: int,
    crop_size: int,
    mean: List[float],
    std: List[float],
) -> List[np.ndarray]:
    """
    Performs normalization and applies required transforms
    to prepare the input frames and returns a list of arrays.
    Specifically the following actions are performed
    1. scale the short side of the frames
    2. center crop the frames to crop_size
    3. perform normalization by subtracting mean and dividing std
    4. sample frames for specified num_frames
    5. sample frames for slow and fast pathways
    """
    inputs = [scale_short_side(size=crop_size, frame=frame) for frame in frames]
    inputs = [center_crop(size=crop_size, frame=frame) for frame in inputs]
    inputs = np.array(inputs).astype(np.float32) / 255
    inputs = normalize(array=inputs, mean=mean, std=std)
    # T H W C -> C T H W
    inputs = inputs.transpose([3, 0, 1, 2])
    # Sample frames for num_frames specified
    indices = np.linspace(0, inputs.shape[1] - 1, num_frames).astype(np.int_)
    inputs = np.take(inputs, indices=indices, axis=1)
    # prepare pathways for the model
    inputs = pack_pathway_output(inputs)
    inputs = [np.expand_dims(inp, 0) for inp in inputs]
    return inputs


def run_inference(
    model: Any,
    video_path: str,
    top_k: int,
    id_to_label_mapping: Dict[str, str],
    num_frames: int,
    sampling_rate: int,
    crop_size: int,
    mean: List[float],
    std: List[float],
) -> List[str]:
    """
    Run inference on the video given by video_path using the given model.
    First, the video is loaded from source. Frames are collected, processed
    and fed to the model. The top top_k predicted class IDs are converted to class
    labels and returned as a list of strings.
    """
    video_cap = cv2.VideoCapture(video_path)
    frames = []
    seq_length = num_frames * sampling_rate
    # get the list of frames from the video
    ret = True
    while ret and len(frames) < seq_length:
        ret, frame = video_cap.read()
        frames.append(frame)
    # prepare the inputs
    inputs = process_inputs(
        frames=frames, num_frames=num_frames, crop_size=crop_size, mean=mean, std=std
    )

    if isinstance(model, ov.CompiledModel):
        # openvino compiled model
        output_blob = model.output(0)
        predictions = model(inputs)[output_blob]
    else:
        # pytorch model
        predictions = model([torch.from_numpy(inp) for inp in inputs])
        predictions = predictions.detach().cpu().numpy()

    def softmax(x):
        return (np.exp(x) / np.exp(x).sum(axis=None))

    # apply activation
    predictions = softmax(predictions)

    # top k predicted class IDs
    topk = 5
    pred_classes = np.argsort(-1 * predictions, axis=1)[:, :topk]

    # Map the predicted classes to the label names
    pred_class_names = [id_to_label_mapping[int(i)] for i in pred_classes[0]]
    return pred_class_names



NUM_FRAMES = 32
SAMPLING_RATE = 2
CROP_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
TOP_K = 5

predictions = run_inference(
    model=model,
    video_path=str(VIDEO_PATH),
    top_k=TOP_K,
    id_to_label_mapping=kinetics_id_to_classname,
    num_frames=NUM_FRAMES,
    sampling_rate=SAMPLING_RATE,
    crop_size=CROP_SIZE,
    mean=MEAN,
    std=STD,
)

print(f"Predicted labels: {', '.join(predictions)}")


ONNX_MODEL_PATH = MODEL_DIR / "slowfast-r50.onnx"
dummy_input = [torch.randn((1, 3, 8, 256, 256)), torch.randn([1, 3, 32, 256, 256])]
torch.onnx.export(
    model=model,
    args=dummy_input,
    f=ONNX_MODEL_PATH,
    export_params=True,
)


model = ov.convert_model(ONNX_MODEL_PATH)
IR_PATH = MODEL_DIR / "slowfast-r50.xml"

# serialize model for saving IR
ov.save_model(model=model, output_model=str(IR_PATH), compress_to_fp16=False)


core = ov.Core()

# read converted model
conv_model = core.read_model(str(IR_PATH))


import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device



# load model on device
compiled_model = core.compile_model(model=conv_model, device_name=device.value)

pred_class_names = run_inference(
    model=compiled_model,
    video_path=str(VIDEO_PATH),
    top_k=TOP_K,
    id_to_label_mapping=kinetics_id_to_classname,
    num_frames=NUM_FRAMES,
    sampling_rate=SAMPLING_RATE,
    crop_size=CROP_SIZE,
    mean=MEAN,
    std=STD,
)
print(f"Predicted labels: {', '.join(pred_class_names)}")

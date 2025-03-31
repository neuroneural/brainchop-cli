from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict
import json
import nibabel as nib
import numpy as np
import time


def convert_keys(torch_state_dict, tiny_state_dict):
    torch_keys = torch_state_dict.keys()
    tiny_keys  = tiny_state_dict.keys()
    new_dict   = {}
    for (f, t) in zip(torch_keys, tiny_keys):
        new_dict[t] = torch_state_dict[f]
    return new_dict

def pretty_print(state_dict):
    """Prints a PyTorch model state dict in a readable format"""
    import re
    for key, tensor in state_dict.items():
        shape_match = re.search(r'\((\d+(?:,\s*\d+)*)\)', str(tensor))
        shape_str = shape_match.group(1) if shape_match else "unknown"
        print(f"{key}: shape=({shape_str})")
    print(f"\nTotal layers: {len(state_dict)}")

def qnormalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = np.quantile(img, qmin)
    qhigh = np.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = np.clip(img, 0, 1)  # Clip the values to be between 0 and 1
    return img

def set_channel_num(config, in_channels, n_classes, channels):
  # input layer
  config["layers"][0]["in_channels"] = in_channels
  config["layers"][0]["out_channels"] = channels
  # output layer
  config["layers"][-1]["in_channels"] = channels
  config["layers"][-1]["out_channels"] = n_classes
  # hidden layers
  for layer in config["layers"][1:-1]:
    layer["in_channels"] = layer["out_channels"] = channels
  return config

def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
  layers = []
  kwargs["kernel_size"] = [kwargs["kernel_size"]]*3
  layers.append(nn.Conv2d(*args, **kwargs))
  if bnorm:
      layers.append(
          nn.GroupNorm(
              num_groups=kwargs["out_channels"],
              num_channels=kwargs["out_channels"],
              affine=False,
          )
      )
    
  relu = lambda x: x.relu()
  gelu = lambda x: x.gelu()
  dropout = lambda x: x.dropout(dropout_p)
  layers.append(gelu if gelu else relu)
  if dropout_p > 0:
    layers.append(dropout)
  return layers

class MeshNet:
  """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""
  def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
    """Init"""
    with open(config_file, "r") as f:
      config = set_channel_num(
        json.load(f), in_channels, n_classes, channels
      )
    if fat is not None:
      chn = int(channels * 1.5)
      if fat in {"i", "io"}:
        config["layers"][0]["out_channels"] = chn
        config["layers"][1]["in_channels"] = chn
      if fat == "io":
        config["layers"][-1]["in_channels"] = chn
        config["layers"][-2]["out_channels"] = chn
      if fat == "b":
        config["layers"][3]["out_channels"] = chn
        config["layers"][4]["in_channels"] = chn
        
    self.model = []
    for block_kwargs in config["layers"][:-1]:  # All but the last layer
      self.model.extend(
        construct_layer(
          dropout_p=config["dropout_p"],
          bnorm=config["bnorm"],
          gelu=config["gelu"],
          **{**block_kwargs, "bias": False}, # middle layers have no bias
        )
      )
    
    # Handle last layer specially - add it to model list
    last_config = config["layers"][-1]
    self.model.append(
      nn.Conv2d(
        last_config["in_channels"],
        last_config["out_channels"],
        kernel_size=[last_config["kernel_size"]]*3,
        padding=last_config["padding"],
        stride=last_config["stride"],
        dilation=last_config["dilation"],
        bias=False # Enable bias in the conv layer
      )
    )
    
  def __call__(self, x):
    # Process all layers except the last one
    for layer in self.model:
      x = layer(x)
    return x


def load_nifti(nifti_path):
    """Load a NIfTI file and return its data as a numpy array"""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.int32)
    # Store affine for later reconstruction
    affine = img.affine
    return data, affine

def save_segmentation(segmentation, affine, output_path):
    """Save the segmentation as a NIfTI file"""
    # Convert to numpy array and get class with highest probability (if multi-class)
    seg_class = segmentation.argmax(1)[0].numpy().astype(np.int32)
    seg_img = nib.Nifti1Image(seg_class, affine)
    nib.save(seg_img, output_path)
    print(f"Segmentation saved to {output_path}")

def run_inference(model, nifti_path, output_path):
    """Load NIfTI data, run inference with model, and save the result"""
    print(f"Loading {nifti_path}...")
    volume_data, affine = load_nifti(nifti_path)
    
    print(f"Volume shape: {volume_data.shape}")
    print("Preprocessing volume...")
    input_tensor = Tensor(qnormalize(volume_data), dtype=dtypes.float).rearrange("... -> 1 1 ...") 
    print("Running inference...")
    start_time = time.time()
    output = model(input_tensor).realize()
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    print("Post-processing and saving result...")
    save_segmentation(output, affine, output_path)


def load_meshnet(model_fn:str, config_fn:str, 
                 in_channels:int = 1, channels:int = 1, out_channels:int = 1):
    # Interpret channel info from config
    model = MeshNet(
        in_channels=in_channels, 
        n_classes=out_channels,
        channels=channels, 
        config_file=config_fn
    )
    state_dict = torch_load(model_fn)
    state_dict = convert_keys(state_dict, nn.state.get_state_dict(model))
    load_state_dict(model, state_dict, strict=True)
    return model
    


if __name__ == "__main__":
    # Paths
    nifti_path = "t1_crop.nii.gz"  # Input NIfTI file
    model_path = "mindgrab.pth"    # Pretrained model
    config_path = "mindgrab.json"  # Model config
    output_path = "segmentation_output.nii.gz"  # Output segmentation
    model = load_meshnet(model_path,config_path,
                         1, 15, 2)
    run_inference(model, nifti_path, output_path)

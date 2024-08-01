import json
import numpy as np
from tinygrad import Tensor 

def normalize(img):
  img = (img - img.min()) / (img.max() - img.min())
  return img

def load_tfjs_model(json_path, bin_path):
  with open(json_path, "r") as f:
    model_spec = json.load(f)

    with open(bin_path, "rb") as f:
      weights_data = np.frombuffer(f.read(), dtype=np.float32)

    return model_spec, weights_data

def create_activation(name):
  activation_map = {
    "relu":         lambda x: x.relu(),
    "elu":          lambda x: x.elu(),
    "sigmoid":      lambda x: x.sigmoid(),
    "tanh":         lambda x: x.tanh(),
    "leaky_relu":   lambda x: x.leakyrelu(),
  }
  return activation_map[name]

def calculate_same_padding(kernel_size, dilation):
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size,) * 3
    if isinstance(dilation, int):
      dilation = (dilation,) * 3

  padding = []
  for k, d in zip(kernel_size, dilation):
    padding.append((k - 1) * d // 2)
  return tuple(padding)


def tinygrad_model(json_path, bin_path, x):

  model_spec, weights_data = load_tfjs_model(json_path, bin_path)
  x = normalize(x)
  weight_index = 0
  in_channels = 1  # Start with 1 input channel
  spec = model_spec["modelTopology"]["model_config"]["config"]["layers"][1:]
  for i, layer in enumerate(spec): # skip input layer
    if layer["class_name"] == "Conv3D":
      config = layer["config"]
      padding = calculate_same_padding(
        config["kernel_size"], config["dilation_rate"]
      )

      in_channels=in_channels,
      out_channels=config["filters"],
      kernel_size=config["kernel_size"],
      stride=config["strides"],
      padding=padding,
      dilation=config["dilation_rate"],

      # Load weights and biases
      k, k, k = kernel_size[0]
      weight_shape = [out_channels[0], in_channels[0], k,k,k]
      # putting the shape into tfjs order
      weight_shape = [weight_shape[i] for i in (2, 3, 4, 1, 0)]
      bias_shape = [out_channels[0]]

      weight_size = np.prod(weight_shape)
      bias_size = np.prod(bias_shape)

      weight = weights_data[
      weight_index : weight_index + weight_size].reshape(weight_shape)
      weight = np.transpose(weight, (4, 3, 0, 1, 2))
      weight_index += weight_size

      bias = weights_data[
      weight_index : weight_index + bias_size].reshape(bias_shape)
      weight_index += bias_size

      weight_data = Tensor(weight.copy())
      bias_data = Tensor(bias.copy())
      x = x.conv2d(
        weight = weight_data,
        bias = bias_data, 
        groups = 1,
        stride = stride[0][0],
        dilation = dilation[0][0],
        padding = padding[0][0]
      )
      in_channels = out_channels[0]

    elif layer["class_name"] == "Activation":
      activation = create_activation(layer["config"]["activation"])
      x = activation(x)
  return x.argmax(1).numpy()[0]

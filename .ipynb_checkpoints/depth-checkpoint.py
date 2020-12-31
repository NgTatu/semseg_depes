from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import depes.networks
from depes.utils import download_model_if_doesnt_exist

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_depth(image_path, max_depth, min_depth, scale, loaded_dict_enc, encoder, depth_decoder):
  input_image = pil.open(image_path).convert('RGB')
  original_width, original_height = input_image.size

  feed_height = loaded_dict_enc['height']
  feed_width = loaded_dict_enc['width']
  input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

  input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0) 
  with torch.no_grad():
    features = encoder(input_image_pytorch)
    outputs = depth_decoder(features)

  disp = outputs[("disp", 0)]
  disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)

  # Saving colormapped depth image
  disp_resized_np = disp_resized.squeeze().cpu().numpy()
  scaled_disp, depth = disp_to_depth(disp_resized_np, min_depth, max_depth)
  real_depth = scale*depth
  return real_depth

if __name__ == "__main__":
  print('ok')
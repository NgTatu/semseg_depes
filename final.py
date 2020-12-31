import torch
import argparse
import os
import numpy as np



from semseg.ptsemseg.models import get_model
from semseg.ptsemseg.loader import get_loader
from semseg.ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True
import cv2


def init_model(size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader("cityscapes")
    loader = data_loader(
        root='/content/drive/MyDrive/data_unzip',
        is_transform=True,
        img_size=eval(size),
        test_mode=True
    )
    n_classes = loader.n_classes
    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return device, model, loader


def test(size, model_path, _input):
    device, model, loader = init_model(size, model_path)
    proc_size = eval(size)
    img_raw, decoded = process_img(_input, proc_size, device, model, loader)
    print(decoded)
    return decoded

def process_img(img_path, size, device, model, loader):
    print("Read Input Image from : {}".format(img_path))

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    return img_resized, decoded


if __name__ == "__main__":
    size = ("540,960")
    model_path = "/content/semseg_depes/semseg_depes/semseg/model/hardnet70_cityscapes_model.pkl"
    _input = "/content/drive/MyDrive/data_unzip/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
    test(size, model_path,_input)

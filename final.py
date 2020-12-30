from __future__ import absolute_import, division, print_function

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader("cityscapes")
    loader = data_loader(
        root='/home/vandung98/Desktop/semseg_depes/data',
        is_transform=True,
        img_size=eval(args.size),
        test_mode=True
    )
    n_classes = loader.n_classes

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(args.model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model, loader

def process_img(img_path, size, model, loader):
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

    # images = img.to(device)
    # outputs = model(images)
    # print("outputs:")
    # print(outputs.shape)
    # print(outputs)
    # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    # print(pred)
    # print(pred.shape)
    decoded = loader.decode_segmap(img)
    # print(decoded)
    # print(decoded.shape)

    return img_resized, decoded

# def init_model_segment(size,model_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_loader = get_loader("cityscapes")
#     loader = data_loader(
#         root='/home/vandung98/Desktop/semseg_depes/data',
#         is_transform=True,
#         img_size=eval(size),
#         test_mode=True
#     )
#     n_classes = loader.n_classes
#     # Setup Model
#     model = get_model({"arch": "hardnet"}, n_classes)
#     state = convert_state_dict(torch.load(model_path, map_location=device)["model_state"])
#     model.load_state_dict(state)
#     return model

def get_segment(input_path, size):
    img_raw, decoded = process_img(input_path, size, device, model, loader)
    return decoded
    
if __name__=='__main__':
    input_path = "/home/vandung98/Desktop/semseg_depes/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png"
    size = (540,960)
    get_segment(input_path,size)
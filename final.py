from semseg.test import process_img

def init_model_segment(size,model_path):
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
    return model

def get_segment(input_path, size, device, model, loader):
    img_raw, decoded = process_img(input_path, size, device, model, loader)
    return decoded
    
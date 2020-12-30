from semseg.test import process_img

def init_model(size,model_path):
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

def test(size,model_path,input,output):
    device, model, loader = init_model(size,model_path)
    proc_size = eval(size)

    if os.path.isfile(input):
        img_raw, decoded = process_img(input, proc_size, device, model, loader)
        blend = np.concatenate((img_raw, decoded), axis=1)
        # print(img_raw)
        # print(decoded)
        # print(blend.shape)
        out_path = os.path.join(output, os.path.basename(input))
        cv2.imwrite("/content/test.png", decoded)
        cv2.imwrite(out_path, blend)

    elif os.path.isdir(input):
        print("Process all image inside : {}".format(input))

        for img_file in os.listdir(input):
            _, ext = os.path.splitext(os.path.basename((img_file)))
            if ext not in [".png", ".jpg"]:
                continue
            img_path = os.path.join(input, img_file)

            img, decoded = process_img(img_path, proc_size, device, model, loader)
            # print(img)
            # print(decoded)
            blend = np.concatenate((img, decoded), axis=1)
            # print(blend)
            out_path = os.path.join(output, os.path.basename(img_file))
            cv2.imwrite(out_path, blend)
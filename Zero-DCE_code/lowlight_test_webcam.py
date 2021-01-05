import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import time
import cv2
import numpy as np
unloader = transforms.ToPILImage()

'''Good blog that explain convert image to tensor etc
https://oldpan.me/archives/pytorch-tensor-image-transform'''

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

def  tensor_to_PIL ( tensor ) :
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def lowlight(data_lowlight):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
    data_lowlight = Image.fromarray(data_lowlight)
    # basewidth = 640
    # wpercent = (basewidth / float(data_lowlight.size[0]))
    # hsize = int((float(data_lowlight.size[1]) * float(wpercent)))
    # data_lowlight = data_lowlight.resize((basewidth, hsize), Image.ANTIALIAS)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(
        torch.load('/media/ivan/Ivan/Final Applied CV/Zero-DCE-master/Zero-DCE_code/snapshots/Epoch99.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    enhanced_image = tensor_to_PIL(enhanced_image)


    end_time = (time.time() - start)
    print(end_time)

    return enhanced_image


if __name__ == '__main__':
    # test_images
    mirror=1
    path="udp://127.0.0.1:8081"
    with torch.no_grad():
        cam = cv2.VideoCapture(path)
        while True:
            ret_val, img = cam.read()
            if mirror:
                img = cv2.flip(img, 1)

            result1 = lowlight(img)
            result1 = np.asarray(result1) #Convert back to OpenCV
            result1 = result1[:, :, ::-1].copy()
            print(result1.shape)

            result2 = simplest_cb(result1, 5)

            cv2.imshow('Original', img) #SHow original Image
            cv2.imshow('Zero DCE', result1) #Show Zero DCE Only
            cv2.imshow('Zero DCE+CB', result2) #Show Zero DCE Only

            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()


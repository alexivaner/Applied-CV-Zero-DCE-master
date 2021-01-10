import torch
import torchvision
import torch.optim
import os

import model

from PIL import Image
import glob
import time
import cv2
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

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


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_lowlight = Image.open(image_path)
    basewidth = 640
    wpercent = (basewidth / float(data_lowlight.size[0]))
    hsize = int((float(data_lowlight.size[1]) * float(wpercent)))
    data_lowlight = data_lowlight.resize((basewidth, hsize), Image.ANTIALIAS)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(
        torch.load('/media/ivan/Ivan/Final Applied CV/Zero-DCE-master/Zero-DCE_code/snapshots/Original.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    path_zero = "/media/ivan/Ivan/Final Applied CV/images/result/test_img/capt_prev.jpg"
    torchvision.utils.save_image(enhanced_image, path_zero)


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = '/media/ivan/Ivan/Final Applied CV/images/original/'

        file_list = os.listdir(filePath)
        print(file_list)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                # image = image
                print(image)
                lowlight(image)
                path_zero="/media/ivan/Ivan/Final Applied CV/images/result/test_img/capt_prev.jpg"
                img = cv2.imread(path_zero)
                out = simplest_cb(img, 5)
                correct_path="/media/ivan/Ivan/Final Applied CV/images/result/test_img/correct_prev.jpg"
                cv2.imwrite(correct_path, out)
                print("Correct written")

                img = cv2.imread(correct_path)
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

                denoise_img_wavelet = np.uint8(denoise_wavelet(img2, multichannel=True, rescale_sigma=True) * 255)
                wavelet_path="/media/ivan/Ivan/Final Applied CV/images/result/test_img/wavelet_prev.jpg"
                cv2.imwrite(wavelet_path, cv2.cvtColor(denoise_img_wavelet, cv2.COLOR_RGB2BGR))
                print("Denoise wavelet written")

                dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 3, 15)
                opencv_path="/media/ivan/Ivan/Final Applied CV/images/result/test_img/opencv_prev.jpg"
                cv2.imwrite(opencv_path, dst)
                print("Denoise opencv written")

import torch
import matplotlib.pyplot as plt
import torchvision
import torch.optim
import os
from torchvision import transforms
import model
import numpy as np
from PIL import Image
import glob
import time
import cv2
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


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
    DCE_net.load_state_dict(torch.load('snapshots/Epoch149.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print(end_time)
    image_path = image_path.replace('test_data', 'result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path


def lowlight_video(image):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    basewidth = 640
    wpercent = (basewidth / float(im_pil.size[0]))
    hsize = int((float(im_pil.size[1]) * float(wpercent)))
    data_lowlight = im_pil.resize((basewidth, hsize), Image.ANTIALIAS)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch149.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    print(end_time)

    enhanced_image = enhanced_image.cpu().detach().numpy()

    enhanced_image_np = enhanced_image[0]
    enhanced_image_tr = np.transpose(enhanced_image_np, (1, 2, 0))

    return np.uint8(enhanced_image_tr*255)


def image_correction(image):
    path = lowlight(image)

    img = cv2.imread(path)
    out = simplest_cb(img, 5)
    correct_path = os.path.splitext(path)[0] + "_correct.jpg"
    cv2.imwrite(correct_path, out)
    print("Correct written")

    img = cv2.imread(correct_path)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

    denoise_img_wavelet = np.uint8(denoise_wavelet(img2, multichannel=True, rescale_sigma=True) * 255)
    denoise_path_wavelet = os.path.splitext(path)[0] + "_correct_denoise_wavelet.jpg"
    cv2.imwrite(denoise_path_wavelet, cv2.cvtColor(denoise_img_wavelet, cv2.COLOR_RGB2BGR))
    print("Denoise wavelet written")

    dst = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 3, 15)
    denoise_path_opencv = os.path.splitext(path)[0] + "_correct_denoise_opencv.jpg"
    cv2.imwrite(denoise_path_opencv, dst)
    print("Denoise opencv written")


def video_correction(image):
    lowlight = cv2.cvtColor(lowlight_video(image), cv2.COLOR_RGB2BGR)
    out = simplest_cb(lowlight, 5)
    dst = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 3, 15)

    return dst


def video_opener(image):
    # Create an object to read
    # from video
    video = cv2.VideoCapture(image)
    print(image)

    filePath=os.path.splitext(image.replace('test_data','result'))[0]
    dirPath=os.path.dirname(filePath)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    frame_count = int(video.get(cv2.CAP_PROP_FPS))


    if (video.isOpened() == False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    basewidth = 640
    wpercent = (basewidth / float(frame_width))
    height = int((float(frame_height * float(wpercent))))

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    fourcc = 0x7634706d
    result = cv2.VideoWriter(filePath+'.avi',
                             fourcc,
                             frame_count, (basewidth,height))

    while (True):
        ret, frame = video.read()

        if ret is True:
            frame = video_correction(frame)
            # cv2.imshow("window",frame)
            result.write(frame)
        else:
            break
        # cv2.waitKey(0)
    # When everything done, release
    # the video capture and video
    # write objects
    video.release()
    result.release()
    print("The video was successfully saved")


if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'data/test_data/Video_tes/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                # image = image
                if image.lower().endswith('.jpg'):
                    image_correction(image)
                    print(image)
                elif (image.lower().endswith(('.mp4', '.mov', '.jpeg'))):
                    video_opener(image)
                    print(image)

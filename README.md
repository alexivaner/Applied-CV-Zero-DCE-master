# Realtime Improvement and Real Time Demo for DSLR Camera : Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

 Final Project for Applied Computer Vision 2020 (Fall 2020) in National Chiao Tung University<br>
 *Best Demo Award in Applied Computer Vision 2020 (Fall 2020)*<br>
 
<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/Photo result.jpg width="1000"> 

 ## Final Demo  Reviewer
[Min-Chun Hu - Multimedia Information System Lab - National Tsing Hua University](http://mislab.cs.nthu.edu.tw/)<br>
[Yi-Ping Chao - Chang Gung University](https://scholar.google.com/citations?user=PWIYaX8AAAAJ&hl=en)<br>
[Norman Wu - Garmin Corporation](https://www.linkedin.com/in/norman-wu-70078b97/?originalSubdomain=tw)<br>

## Lecturer
[Wen-Huang Cheng - AIMM Lab - NCTU](http://aimmlab.nctu.edu.tw/member.html)<br>


## Our Team
[Farhan Tandia](https://github.com/farhantandia)<br>
[Ivan Surya Hutomo](https://github.com/alexivaner)<br>
[Martin Dominikus](https://github.com/mdtjan)<br>

 
# Highlight
We improve previous paper by implement it for real-time demo in DSLR camera and also add color correction and denoising to make the result is more suitable for real implementation. We continue the code from previous works:<br>
* Find more about Zero-DCE click [here](https://li-chongyi.github.io/Proj_Zero-DCE.html)<br>
* Original Zero-DCE Repository click  [here](https://github.com/Li-Chongyi/Zero-DCE)<br>
* Paper CVPR-2020 Zero-DCE Paper click [here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)<br>

# Abstract
Zero-DCE is a novel method to do light enhancement in an image. We could obtain do light enhancement in image while keep maintain the detail and preserves quality of the image. Enhancement may also recover the object detection or recognition in the low-light area. Zero-DCE does not require any paired or unpaired data in the training process as in existing CNN-Based because It using non-reference loss functions. It is said that Zero-DCE supersedes State-of-the-Arts. It is capable of processing images in real-time (about 500 FPS for images of size 640x480x3 on GPU) hence we would to do real implementation of Zero-DCE, in this case implement it using DSLR camera and some input images and videos.

# Goals and Proposed Method
<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/Our%20Improvement.jpg" width="1000"><br>

* Previous Zero-DCE not really has correct color, we aim to do color correction for Zero-DCE.
* If the input image already has some noise, Zero-DCE result will also has some noise, hence we would to improve the result by doing denoising at the output.
* We add second part of SICE dataset and also our own dataset then retrain previous Zero-DCE model to get improvement in the result.
<br><br>All of improvement above is to make sure that we could use Zero-DCE for real-purpose, hence we also combine our demo with DSLR camera using NodeRED, you could see our proposed method below:<br>

<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/What%20we%20do.jpg" width="1000"><br>

# Our Result

## Real Time Inference using DSLR Camera:
<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/Screenshot from 2021-01-20 20-12-33.png" width="1000"> 

## Images
* ED = Extended Dataset (With our dataset + SICE Dataset part 2)
* CC = Color Correction (Click [here](https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html#:~:text=In%20order%20to%20deal%20with,Values%20around%200.01%20are%20typical.) for more detail)
* D = Denoise (Click [here](https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html) for more detail) <br>


| Original Images | ZeroDCE | ZeroDCE+ED+CC+D (Ours) |
| ------------- | ------------- | ------------- |
| <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_170729Ori.jpg" width="400"> | <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_170729.jpg" width="400"><br> |  <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_170729_correct_denoise_opencv.jpg" width="400"><br> |
| <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/3Ori.jpg" width="400"> | <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/3.jpg" width="400"><br> |  <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/3_correct_denoise_opencv.jpg" width="400"><br> |
| <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20201228_182159Ori.jpg" width="400"> | <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20201228_182159.jpg" width="400"><br> |  <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20201228_182159_correct_denoise_opencv.jpg" width="400"><br> |
| <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_173105Ori.jpg" width="400"> | <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_173105.jpg" width="400"><br> |  <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/P_20191215_173105_correct_denoise_opencv.jpg" width="400"><br> |


## Videos
<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/video_compare.gif" width="400">

# How to Run
You could run the program by following steps::<br>
### Clone the repository:<br>
 `git clone https://github.com/alexivaner/Applied-CV-Zero-DCE-master/` <br><br>
 
### Download the and Dataset:
* Download dataset [here](https://drive.google.com/file/d/1S2yJEgUKCMJxLTZqcQwoxpuuvaA-OD93/view?usp=sharing)<br>
Our dataset is originally taken from SICE DATASET, if you want to know more, you can click [here](https://github.com/csjcai/SICE)<br>
We also take our own dataset, We add  around 190 Photos, with 38 Different Scenes inside NCTU Campus. Photos are taken using Nikon D5300 and Fujifilm XT-100
* After download the dataset, extract the "data" inside dataset to Zero-DCE-master/Zero-DCE_code dolder, so the structure of folder will be like this:
  
<pre>
└── <font color="#3465A4"><b>Zerp-DCE-master</b></font>
    ├── <font color="#3465A4"><b>Zero-DCE_code</b></font>
    │   ├── data
    │   ├── snapshots
    │   ├── ...
    ├── <font color="#3465A4"><b>Tools</b></font>
    │   ├── Crop Image.ipynb
    ├── <font color="#3465A4"><b>....</b></font>

</pre>
 
 ### Install Environment using Conda
 We already provide environment file in this repository to make sure the installation is easy: <br>
 * Go to Zero-DCE-master folder: <br>
 `cd Zero-DCE-master` <br>
 * Make sure you have anaconda in your pc
* Run this command to install environment using conda: <br>
 `conda env create -f Zero-DECE-master.yml` <br>
 
 
### Inference and Evaluate
#### Webcam
* Go to Zero-DCE_code folder: <br>
 `cd Zero-DCE_code` <br>
* Run lowlight_test_webcam.py: <br>
 `python lowlight_test_webcam.py` <br>
 
#### Testing Image/VIdeo
* Go to Zero-DCE_code folder: <br>
 `cd Zero-DCE_code` <br>
* Do not forget to edit the path in line 180 of lowlight_test_file.py
* Run lowlight_test_file.py: <br>
 `python llowlight_test_file.py` <br>
 
#### Testing using DSLR Camera
If you want to inference using DSLR Camera, make sure you install the requirement below:
* Node-RED (click [here](https://nodered.org/docs/getting-started/local) for installation)
* gPhoto2 Library (click [here](https://zoomadmin.com/HowToInstall/UbuntuPackage/gphoto2) for installation)

After Node-RED Installed:
* Run `node-red-start` in the terminal
* Visit http://127.0.0.1:1880/ in your browser
* Click hamburger menu in the top-right corner, then go to manage palette, click palette, click install
* Find and Install Node-RED dashboard and blynk

After everything is ready, you need to set httpstatic in Node-RED so our Node-RED can access our folder:
* Change configuration in setting.js, for me the command will be like this: <br>
`sudo nano /home/ivan/.node-red/settings.js`, change "ivan" with your username
* Uncomment httpStatic and replace wtih path for "images" folder in our repos, for example:<br>
`
  // When httpAdminRoot is used to move the UI to a different root path, the
    // following property can be used to identify a directory of static content
    // that should be served at http://localhost:1880/.
    httpStatic: '/media/ivan/Ivan/Final Applied CV/Zero-DCE-master/images'
`
* Save the setting ctrl+x, press y, and press enter if you are using nano.
* Restart Node-RED by run `node-red-stop` then `node-red-start`

After set httpStatic, its time to import the flow:
* Visit http://127.0.0.1:1880/ in your browser
* Click hamburger menu in the top-right corner, then click import
* Click `Select file to import`, choose `nodered_flow 14 Januari 2021.json`, then click import
* Click Deploy in the right corner of NodeRED

Now you should already see the interface of NodeRED-Dashboard with your browser:
* Visit http://127.0.0.1:1880/ui in your browser and you should see some interfaces
* Visit http://127.0.0.1:1880/ in your browser
* You need to make sure all the path in 'capture camera','enhance image','watch for pic', and every 'timestamp' already refer to your correct path. Do not forget to change python path based on your anaconda environment path.
* Do not forget to click 'Deploy' if you change something.

After everything already setup:
* Connect your DSLR Camera with USB Cable
* Visit http://127.0.0.1:1880/ui in your browser and you should see some interfaces
* Click 'Capture camera' and see if your camera already give some respond.



### Training Zero-DCE
* Go to Zero-DCE_code folder: <br>
 `cd Zero-DCE_code` <br>
* Do not forget to edit the path from lowlight_train.py
* Run lowlight_train.py: <br>
 `python lowlight_train` <br>


## Full Proposal
Please download our full proposal here:<br>
[Full Proposal](https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/Final%20Applied%20CV%20Proposal.pdf)

## Full Final Explanation Report
Please download our full final report here:<br>
[Full Report](https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/Final%20Applied%20CV%20Final%20Result.pdf)

## Disclaimer
Please cite us as authors, our GitHub, and Original's Zero DCE repository if you plan to use this as your next research or any paper.<br>

# Reference
<pre>
Guo, Chunle, et al. "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

Zer-DCE Github Page - https://li-chongyi.github.io/Proj_Zero-DCE.html<br>

Color Balance - https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html<br>

OpenCV Denoise - https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html<br>
<br>
</pre>

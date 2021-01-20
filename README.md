# Realtime Improvement and Real Time Demo for DSLR Camera : Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

 Final Project for Applied Computer Vision 2020 (Fall 2020) in National Chiao Tung University<br>
 *Best Demo Award in Applied Computer Vision 2020 (Fall 2020)*<br>
 
# Highlight
We improve previous paper by implement it for real-time demo in DSLR camera and also add color correction and denoising to make the result is more suitable for real implementation. We continue the code from previous works:<br>
* Find more about Zero-DCE click [here](https://li-chongyi.github.io/Proj_Zero-DCE.html)<br>
* Original Zero-DCE Repository click  [here](https://github.com/Li-Chongyi/Zero-DCE)<br>
* Paper CVPR-2020 Zero-DCE Paper click [here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)<br>

          
 <img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/Photo%20result.jpg" width="800"><br>

 ## Final Demo  Reviewer
[Min-Chun Hu - Multimedia Information System Lab - National Tsing Hua University](http://mislab.cs.nthu.edu.tw/)<br>
[Yi-Ping Chao - Chang Gung University](https://scholar.google.com/citations?user=PWIYaX8AAAAJ&hl=en)<br>
[Norman Wu - Garmin Corporation](https://www.linkedin.com/in/norman-wu-70078b97/?originalSubdomain=tw)<br>

## Lecturer
[Wen-Huang Cheng - AIMM Lab - NCTU](http://aimmlab.nctu.edu.tw/member.html)<br>


## Our Team
[Farhan Tandia](https://github.com/farhantandia)<br>
[Ivan Surya Hutomo](https://github.com/alexivaner)<br>
[Martin Dominikus]()<br>


# Abstract
Zero-DCE is a novel method to do light enhancement in an image. We could obtain do light enhancement in image while keep maintain the detail and preserves quality of the image. Enhancement may also recover the object detection or recognition in the low-light area. Zero-DCE does not require any paired or unpaired data in the training process as in existing CNN-Based because It using non-reference loss functions. It is said that Zero-DCE supersedes State-of-the-Arts. It is capable of processing images in real-time (about 500 FPS for images of size 640x480x3 on GPU) hence we would to do real implementation of Zero-DCE, in this case implement it using DSLR camera and some input images and videos.

# Goals and Proposed Method
<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/Our%20Improvement.jpg" width="1000"><br>

* Previous Zero-DCE not really has correct color, we aim to do color correction for Zero-DCE.
* If the input image already has some noise, Zero-DCE result will also has some noise, hence we would to improve the result by doing denoising at the output.
* We add second part of SICE dataset and also our own dataset then retrain previous Zero-DCE model to get improvement in the result.
<br><br>All of improvement above is to make sure that we could use Zero-DCE for real-purpose, hence we also combine our demo with DSLR camera using NodeRED, you could see our proposed method below:<br>

<img src="https://github.com/alexivaner/Applied-CV-Zero-DCE-master/raw/master/readme_source/What%20we%20do.jpg" width="1000"><br>





### Our Result (Green Line)
We could see that our result surpassed previous method a lot in Low SNR, from under 20% to more than 70% (We could see our result in green line surpassed baseline in Low SNR Signal) <br>
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Comparison%202.gif" width="600"><br><br>

#### Comparison in Confussion Matrices:
We could see that we got very good confussion matrices even in the Low SNR Signal
<img src="https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Comparison.gif" width="600"><br>

# How to Run
You could run the program by following steps::<br>
### Clone the repository:<br>
 `git clone https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification.git` <br><br>
 
### Download the Weight and Dataset:
* Download weights [here](https://drive.google.com/drive/folders/1RIjrZaKJW8oCLbd5LANvTqemk8f-1uWK?usp=sharing) <br>
* Extract all weights to "Submission" folder <br>
* Download extracted dataset [here](https://drive.google.com/file/d/1gUPDlvPqCnb_C4k2h3st0JV9p_sSvaiI/view?usp=sharing)<br>
Our dataset is originally taken from DEEPSIG DATASET: RADIOML 2018.01A (NEW), if you want to know more, you can click [here](https://www.deepsig.ai/datasets)<br>
* Create new folder and name it "ExtractDataset", extract all the dataset and put on that folder, the folder structure will be like below:
  
<pre>
└── <font color="#3465A4"><b>Deep-Learning-Based-Radio-Signal-Classification</b></font>
    ├── <font color="#3465A4"><b>Submission</b></font>
    │   ├── resnet_model_mix.h5
    │   ├── trafo_model.data-00000-of-00001
    │   ├── trafo_model.index
    │   ├── ...
    ├── <font color="#3465A4"><b>ExtractDataset</b></font>
    │   ├── part0.h5
    │   ├── part1.h5
    │   ├── part2.h5
    │   ├── ....
</pre>
 
### Inference and Evaluate
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Evaluate-Benchmark.ipnyb": <br>
 
### Training Resnet Modified for High SNR Signal
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Classification-proposed-model-resnet-modified-highest.ipynb": <br>

### Training Transformer Model for Low SNR Signal
* Go to Submission folder: <br>
 `cd Submission` <br>
* Run Jupyter Notebook: <br>
 `jupyter notebook` <br>
* Open "Classification-proposed-model-transformer-low.ipynb": <br>

## Full Proposal
Please download our full proposal here:<br>
[Full Proposal](https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Proposal/Proposal_Team3_Farhan%20Tandia_Ivan%20Surya%20H.pdf)

## Full Final Explanation Report
Please download our full final report here:<br>
[Full Report](https://github.com/alexivaner/Deep-Learning-Based-Radio-Signal-Classification/raw/main/Submission/Final/Final_Team13_Farhan%20Tandia_Ivan%20Surya%20H.pdf)

## Disclaimer
Please cite us as autho, our GitHub, and Original's Zero DCE repository if you plan to use this as your next research or any paper.<br>



# Reference
<pre>
T. J. O’Shea, T. Roy and T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," in IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, pp. 168-179, Feb. 2018, doi: 10.1109/JSTSP.2018.2797022.<br>

Harper, Clayton A., et al. "Enhanced Automatic Modulation Classification using Deep Convolutional Latent Space Pooling." ASILOMAR Conference on Signals, Systems, and Computers.  2020. <br>

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł. & Polosukhin, I. (2017), Attention is all you need, in 'Advances in Neural Information Processing Systems' , pp. 5998--6008 . <br>

J. Uppal, M. Hegarty, W. Haftel, P. A. Sallee, H. Brown Cribbs and H. H. Huang, "High-Performance Deep Learning Classification for Radio Signals," 2019 53rd Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, 2019, pp. 1026-1029, doi: 10.1109/IEEECONF44664.2019.9048897. <br>

S. Huang et al., "Automatic Modulation Classification Using Compressive Convolutional Neural Network," in IEEE Access, vol. 7, pp. 79636-79643, 2019, DOI: 10.1109/ACCESS.2019.2921988. <br>

Huynh-The, Thien & Hua, Cam-Hao & Pham, Quoc-Viet & Kim, Dong-Seong. (2020). MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification. IEEE Communications Letters. 24. 811-815. 10.1109/LCOMM.2020.2968030. <br>
</pre>

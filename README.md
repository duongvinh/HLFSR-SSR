# Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information
This repository contains official pytorch implementation of "Light Field Image Super-Resolution Network via Joint Spatio-Angular and Epipolar Information" submitted in IEEE Transactions on Computational Imaging 2022, by Vinh Van Duong, Thuc Nguyen Huu, Jonghoon Yim, and Byeungwoo Jeon.

## News
[2022-12-26]: we have updated the pre-trained models and codes.


## Results
We share the pre-trained models and the SR LF images generated by our HLFSR-C32 and HLFSR-C64 model on all the 5 datasets for 2x and 4x SR, which are avaliable at https://drive.google.com/drive/folders/1SaTT3iP4GruKcome8r97Y6y54Nrj4gc5

## Code
### Dependencies
* Python 3.6
* Pyorch 1.3.1 + torchvision 0.4.2 + cuda 92
* Matlab

### Dataset
We use the processed data by [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet](https://github.com/YingqianWang/LF-DFnet).

### Prepare Training and Test Data
* To generate the training data, please first download the five datasets and run:
  ```matlab
  GenerateTrainingData_HLFSR.m
* To generate the test data, run:
  ```matlab
  GenerateDataForTest_HLFSR.m
### Train
* Run:
  ```python
  python train_HLFSR.py  --angRes 5 --upscale_factor 4 --channels 64  --crop_test_method 3
### Test
* Run:
  ```python
  python test_HLFSR.py --angRes 5 --upscale_factor 4 --channels 64  --crop_test_method 3 --model_path [pre-trained dir]
  
[Important note]: For our HLFSR method, the performance is following “the larger image patch size is the better”. For example, if we keep the whole image as an input of our network (i.e., crop_test_method is fixed equal to 1), it can be achieved the best performance. This is because our proposed network components require an adequate size of an input image to better exploit the pixel correlations in a larger receptive field. To get the same performance as reported in our paper, we need to set the default crop_test_method equal to 3.

<p align="center"> <img src="https://github.com/duongvinh/HLFSR-SSR/blob/main/Figs/CroppedImageMethods.JPG" width="50%"> </p>
  
### Visual Results
* To merge the Y, Cb, Cr channels, run:
  ```matlab
  GenerateResultImages.m
  
  
## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{
  title={Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information},
  author={Vinh Van Duong, Thuc Nguyen Huu, Jonghoon Yim, and Byeungwoo Jeon},
  journal={submitted to IEEE Transactions on Computational Imaging},
  year={2022},
  publisher={IEEE}
}
```
```Citation
@InProceedings{LF-InterNet,
  author    = {Wang, Yingqian and Wang, Longguang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
  title     = {Spatial-Angular Interaction for Light Field Image Super-Resolution},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {290-308},
  year      = {2020},
}

```Citation
@article{LF-DFnet,
  author  = {Wang, Yingqian and Yang, Jungang and Wang, Longguang and Ying, Xinyi and Wu, Tianhao and An, Wei and Guo, Yulan},
  title   = {Light Field Image Super-Resolution Using Deformable Convolution},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30),
  pages   = {1057-1071},
  year    = {2021},
}

```
## Acknowledgement
Our work and implementations are inspired and based on the following projects: <br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br> 
We sincerely thank the authors for sharing their code and amazing research work!

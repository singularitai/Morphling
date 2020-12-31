# Morphling - Text to human pipeline.
[![](http://img.youtube.com/vi/VxrtbWqwyUk/0.jpg)](http://www.youtube.com/watch?v=VxrtbWqwyUk "Creation of this tool")
A Machine Learning pipeline that goes from text to fully animated faces and voiced faces!

# Video Tool Use Guide

TODO

# Requirements

## Main Requirements
CUDA Enabled GPU

Python > 3

FFMPEG (Ensure it is in path)

## PIP Modules
```python
pip3 install -r requirements.txt
```

## Useage
```python
python3 interface.py
```
# Downloads
A complete downloadable version of this project found on google drive: https://drive.google.com/drive/folders/1uNL7wzCTbG7opHDvW61ly_N8Oa4LTSxy?usp=sharing

## Models

### Drive download
All models can be found on google drive. They should be placed in the models folder.: https://drive.google.com/drive/u/2/folders/1LVEHsGlU5yipw6boRv1ocYfWcPaCi1ox

### Seperate Downloads
These should be placed in the 'models' folder:
	Vox-adv-cpk.pth.tar OR vox-cpk.pth.tar FROM : https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH or https://yadi.sk/d/lEw8uRm140L_eQ 
	alternatively you can use the avatarify version from https://www.dropbox.com/s/t7h24l6wx9vreto/vox-adv-cpk.pth.tar?dl=0 or https://yadi.sk/d/M0FWpz2ExBfgAA or https://drive.google.com/file/d/1coUCdyRXDbpWnEkA99NLNY60mb9dQ_n3/view
	libritts - https://drive.google.com/uc?id=1KhJcPawFgmfvwV7tQAOeC253rYstLrs8 
	waveglow - https://drive.google.com/uc?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF 
	lipgan - https://drive.google.com/uc?id=1DtXY5Ei_V6QjrLwfe7YDrmbSCDu6iru1
	face detector: http://dlib.net/files/mmod_human_face_detector.dat.bz2 
	
## This work has adapted notebooks:
1) Flowtron - https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/NVidia_Flowtron_Waveglow.ipynb
2) LipGan - https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/LipGAN.ipynb#scrollTo=ktXeABjLYb70
3) First order motion - https://colab.research.google.com/github/AliaksandrSiarohin/first-order-model/blob/master/demo.ipynb

## This makes use of several technologies
Each of these uses a different license. Be sure to check them out for more info.
1) Flowtron/Waveglow - https://github.com/NVIDIA/flowtron
2) LipGan - https://github.com/Rudrabha/LipGAN
3) First Order Motion - https://github.com/AliaksandrSiarohin/first-order-model
4) ISR - https://idealo.github.io/image-super-resolution/
5) Stylegan2-Pytorch - https://github.com/Tetratrio/stylegan2_pytorch

# License
This repo is a melting pot of different licenses, including GPL and MIT licenses.

My own contributions are under the 'WTF Public License'


```
            DO WHAT THE F*CK YOU WANT TO PUBLIC LICENCE
                    Version 3.1, July 2019

 by Sam Hocevar <sam@hocevar.net>
    theiostream <matoe@matoe.co.cc>
    dtf         <wtfpl@dtf.wtf>

            DO WHAT THE F*CK YOU WANT TO PUBLIC LICENCE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE F*CK YOU WANT TO.
  
  
```





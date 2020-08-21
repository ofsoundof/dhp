## Dataset structure

    ├── super_resolution
    │   ├── DIV2K
    │   │   ├── DIV2K_train_HR
    │   │   ├── DIV2K_train_LR_bicubic
    │   │   │   ├── X2
    │   │   │   ├── X3
    │   │   │   ├── X4
    │   │   ├── DIV2K_valid_HR
    │   │   ├── DIV2K_valid_LR_bicubic
    │   │   │   ├── X2
    │   │   │   ├── X3
    │   │   │   └── X4
    │   │   └──
    │   ├── benchmark
    │   │   ├── Set5
    │   │   ├── Set14
    │   │   ├── B100
    │   │   ├── Urban100
    │   │   └──
    │   └──
    ├── denoise
    │   ├── DIV2KGRAY
    │   │   ├── Train_HR
    │   │   └──
    │   ├── BenchmarkDenoise
    │   │   ├── DenoiseSet68
    │   │   │   ├── bin
    │   │   │   └──
    │   │   └──
    │   └──
    └──
    
## Test
1. Prepare image super-resolution and denoising dataset.

   i.   Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) training and validation images. Please download the low-resolution images in the NTIRE-2017 challenge. 

   ii.  Download super-resolution (Set5, Set14, B100, and Urban100) benchmark and denoising benchmark (Set68) from [GoogleDrive](https://drive.google.com/file/d/1OXgs2ap0aDymRgSp6uBBVpg6SY6Bpduo/view?usp=sharing) or [Dropbox]().
   
   iii. Prepare the folder structure like the one above.

```

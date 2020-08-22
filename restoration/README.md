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

   i.   Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) training and validation images in the NTIRE-2017 challenge. 

   ii.  Download super-resolution (Set5, Set14, B100, and Urban100) benchmark and denoising benchmark (Set68) from [GoogleDrive](https://drive.google.com/file/d/1OXgs2ap0aDymRgSp6uBBVpg6SY6Bpduo/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/uo75nf4yw5gyk9p/restoration_dataset.zip?dl=0).
   
   iii. Prepare the folder structure like the one above.

2. Download the compressed models form [Google Drive](https://drive.google.com/file/d/1I7S41V_CPHqBepu8lR-Lh8Km-joxnfZ-/view?usp=sharing) or [Dropbox](https://www.dropbox.com/s/6fivh4kaokx6m2s/model_zoo_restoration.zip?dl=0).

3. `cd ./scripts/dhp/`

4. Test the compressed DnCNN, UNet, SRResNet, and EDSR models. 

    ```bash
            bash test_dncnn.sh
            bash test_unet.sh
            bash test_srresnet.sh
            bash test_edsr.sh
    ```

5. Be sure to change the `--pretrain`, `--dir_data`, `--dir_save` directories.

## Train

1. `cd ./scripts/dhp/`

2. Run the code to compress DnCNN, UNet, SRResNet, and EDSR networks. 

	```bash
		bash dhp_dncnn.sh
		bash dhp_unet.sh
		bash dhp_srresnet.sh
		bash dhp_edsr.sh
	```

3. Be sure to change the `--pretrain`, `--dir_data`, `--dir_save` directories.


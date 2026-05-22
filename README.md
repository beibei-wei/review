# Adaptability Improvement of Visual Mamba for Multimodal Image Fusion: A Survey

## Recommended Environment
- torch==2.7.1+cu118
- causal-conv1d==1.5.4   If necessary
- mamba-ssm==2.3.0       If necessary
## Download the model from the original author.
[IASSF        model](https://github.com/fangjiaqi0909/IASSF).  
[EMMA         model](https://github.com/Zhaozixiang1228/MMIF-EMMA).  
[T2EA         model](https://github.com/MysterYxby/T2EA).  
[DCEvo        model](https://github.com/Beate-Suy-Zhang/DCEvo).  
[CDDFuse      model](https://github.com/Zhaozixiang1228/MMIF-CDDFuse).  
[ITFuse       model](https://github.com/tthinking/ITFuse).  
[S4Fusion     model](https://github.com/zipper112/S4Fusion).  
[CAWM-Mamba   model](https://github.com/Feecuin/CAWM-Mamba).  
[WaveletMamba model](https://github.com/Lmmh058/W-Mamba).  
[MKDFusion    model](https://github.com/SEU-ZYC/MKDFusion).  
[SFMFusion    model](https://github.com/Namn23/ISFM).  
[F2Fusion     model](https://github.com/lrh-1994/F2Fusion).  
And you can download the dataset [MSRS](https://github.com/Linfeng-Tang/MSRS).

## File structure
### The fused images are saved in the "data" folder, named after the method name. A few sample images have been pre-stored. Please obtain the relevant code from the original author and run it for the complete results.
```
┬─ data  (origin date and fused date)
│   ├─ origin_vi
│   │   ├─ 00004N.png
│   │   └─ ... (image name)
│   └─ origin_ir 
│   │   └─ ... 
│   └─ CDDFuse (Method name)
│   │    └─ ...
│   └─ ... 
└─ param_cal
│    ├─ CDDFuse_cal.py (based on Method name)
│    │─ DCEvo_cal.py 
│    └─ ... 
│─cal_evatuator.py (Calculation of evaluation indicators)
│─visual.py        (Visualized code)
```
## Evaluation index calculation
Set the storage address and method name for the image to be calculated in "cal_evatuator.py".
## Visualization result output
Set the parameters such as image name, address, and zoom area in "visual.py".
## Parameter quantity calculation
The calculation codes for parameters quantity, GFLOPs, average inference time and FPS are located in "param_cal".
After downloading the corresponding model, place the corresponding calculation program in the main directory and run it.
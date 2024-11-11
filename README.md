# 🌅 DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation

[![arXiv](https://img.shields.io/badge/Arxiv-2410.13726-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.13726)
[![Demo Page](https://img.shields.io/badge/Demo_Page-blue)](https://hanbo-cheng.github.io/DAWN/)
[![zhihu](https://img.shields.io/badge/知乎-0079FF.svg?logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/2253009511)
 <a href='https://huggingface.co/Hanbo-Cheng/DAWN'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>


[中文文档](README_CN.md)
<p align="center">
<img src="structure_img\ifferent-styles-at-higher-resolution.gif" width=600>
</p>


<h5 align="center"> 😊 Please give us a star ⭐ to support us for continous update 😊  </h5>

## News
* ```2024.10.14``` 🔥 We release the [Demo page](https://hanbo-cheng.github.io/DAWN/).
* ```2024.10.18``` 🔥 We release the paper [DAWN](https://arxiv.org/abs/2410.13726).
* ```2024.10.21``` 🔥 We update the Chinese introduction [](https://zhuanlan.zhihu.com/p/2253009511).
* ```2024.11.7``` 🔥🔥 We realse the pretrained model on [hugging face](https://huggingface.co/Hanbo-Cheng/DAWN).
* ```2024.11.9``` 🔥🔥🔥 We realse the inference code. We sincerely invite you to experience our model. 😊
## TODO list:
- [x]  release the inference code
- [x]  release the pretrained model of **128*128**
- [x]  release the pretrained model of **256*256** 
- [ ] release the test code for HDTF dataset
- [ ] in progress ...


## Equipment Requirements

With our VRAM-oriented optimized [code](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test_local_opt.py), the maximum length of video that can be generated is **linearly related** to the size of the GPU VRAM. Larger VRAM produce longer videos.
- To generate **128*128** video, we recommend using a GPU with **12GB** or more VRAM. This can at least generate video of approximately **400 frames**.
- To generate **256*256** video, we recommend using a GPU with **24GB** or more VRAM. This can at least generate video of approximately **200 frames**.

PS: Although optimized [code](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test_local_opt.py) can improve VRAM utilization, it currently sacrifices inference speed due to incomplete optimization of local attention. We are actively working on this issue, and if you have a better solution, we welcome your PR. If you wish to achieve faster inference speeds, you can use [unoptimized code](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test.py), but this will increase VRAM usage (O(n²) spatial complexity).
## Methodology
### The overall structure of DAWN:
<p align="center">
<img src="structure_img\pipeline.png" width=600 alt="framework"/>
</p>


## Environment
We highly recommend to try DAWN on linux platform. Runing on windows may produce some rubbish files need to be deleted manually and requires additional effort for the deployment of the 3DDFA repository (our `extract_init_states` folder) [comment](https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173).

1. set up the conda environment
```
conda create -n DAWN python=3.8
conda activate DAWN
pip install -r requirements.txt
conda create -n 3DDFA python=3.8
conda activate 3DDFA
pip install -r requirements_3ddfa.txt
```

1. Follow the [readme](extract_init_states\readme.md) and [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) to set up the 3DDFA environment.
 

## Inference

Since our model **is trained only on the HDTF dataset** and has few parameters, in order to ensure the best driving effect, please provide examples of :
- standard human photos as much as possible, try not to wear hats or large headgear
- ensure a clear boundary between the background and the subject
- have the face occupying the main position in the image.

The preparation for inference:
1. Download the pretrain checkpoint from [hugging face](https://huggingface.co/Hanbo-Cheng/DAWN). Create the `./pretrain_models` directory and put the checkpoint file into it.
   
2. Changing the path in  `run_ood_test\run_DM_v0_df_test_128_both_pose_blink.sh` or `run_ood_test\run_DM_v0_df_test_256_1.sh`. Infill the `image_path`, `audio_path` and `cache_path`. The `run_ood_test\run_DM_v0_df_test_128_both_pose_blink.sh` is used to perform inference on 128 * 128 images and `run_ood_test\run_DM_v0_df_test_256_1.sh` is used to perform inference on 256 * 256 images.
   
3. Using `bash xxxx.sh` to run the script.

### About PBNet
We provide two PBNet checkpoint: 1. generating both blink and pose together 2. generating blink and pose seperately (script end with "seperate_pose_blink"). According to the quantitative results, these two method has similar performance. 
   


This code is tested on internal server of company and my Windows 11 PC. There might be some minor problems due to the difference of the equipment. Please feel free to leave issues or PR if you encounter some promblems, we are glad to help!

**For testing on datasets**: If you wish to test the performance of DAWN on datasets, we recommend to warp our code and process data in batches for each step (including extracting the initial states, audio embedding, inference of PBNet, inference of A2V-FDM). Reloading the model repeatedly will make your testing efficiency very low. We also plan to release the inference code for HDTF dataset in the future.

## Citing DAWN
If you wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX
@misc{dawn2024,
      title={DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation}, 
      author={Hanbo Cheng and Limin Lin and Chenyu Liu and Pengcheng Xia and Pengfei Hu and Jiefeng Ma and Jun Du and Jia Pan},
      year={2024},
      eprint={2410.13726},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.13726}, 
}
```
## Acknowledgement

[Limin Lin](https://github.com/LiminLin0) and [Hanbo Cheng](https://github.com/Hanbo-Cheng) contributed equally to the project.

Thank you to the authors of [Diffused Heads](https://github.com/MStypulkowski/diffused-heads) for assisting us in reproducing their work! We also extend our gratitude to the authors of [MRAA](https://github.com/snap-research/articulated-animation), [LFDM](https://github.com/snap-research/articulated-animation), [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) and [ACTOR](https://github.com/Mathux/ACTOR) for their contributions to the open-source community. Lastly, we thank our mentors and co-authors for their continuous support in our research work!


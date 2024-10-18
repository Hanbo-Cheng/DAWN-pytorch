# 🌅 DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation

[![arXiv](https://img.shields.io/badge/Arxiv-2410.13726-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.13726)
[![Demo Page](https://img.shields.io/badge/Demo_Page-blue)](https://hanbo-cheng.github.io/DAWN/)

<p align="center">
<img src="structure_img\ifferent-styles-at-higher-resolution.gif" width=600>
</p>

<h5 align="center"> 😊 Please give us a star ⭐ to support us for continous upate 😊  </h5>

## News
* ```2024.10.14``` 🔥 We release the [Demo page](https://hanbo-cheng.github.io/DAWN/).
* ```2024.10.18``` 🔥 We release the paper [DAWN](https://arxiv.org/abs/2410.13726).

## TODO list:
- [ ]  release the inference code
- [ ]  release the pretrained model of **128*128**
- [ ]  release the pretrained model of **256*256**
- [ ] in progress ...


## Equipment Requirements

With our VRAM-oriented optimized code, the maximum length of video that can be generated is **linearly related** to the size of the GPU VRAM. Larger VRAM produce longer videos.
- To generate **128*128** video, we recommend using a GPU with **12GB** or more VRAM. This can at least generate video of approximately **400 frames**.
- To generate **256*256** video, we recommend using a GPU with **24GB** or more VRAM. This can at least generate video of approximately **200 frames**.

PS: Although optimized code can improve VRAM utilization, it currently sacrifices inference speed due to incomplete optimization of local attention. We are actively working on this issue, and if you have a better solution, we welcome your PR. If you wish to achieve faster inference speeds, you can use unoptimized code, but this will increase VRAM usage (O(n²) space complexity).
## Methodology
### The overall structure of DAWN:
<p align="center">
<img src="structure_img\pipeline.png" width=600 alt="framework"/>
</p>


## In Progress ...




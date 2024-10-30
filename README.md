# 🌅 DAWN: Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation

[![arXiv](https://img.shields.io/badge/Arxiv-2410.13726-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.13726)
[![Demo Page](https://img.shields.io/badge/Demo_Page-blue)](https://hanbo-cheng.github.io/DAWN/)
[![zhihu](https://img.shields.io/badge/知乎-0079FF.svg?logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/2253009511)

<p align="center">
<img src="structure_img\ifferent-styles-at-higher-resolution.gif" width=600>
</p>

<h5 align="center"> 😊 Please give us a star ⭐ to support us for continous update 😊  </h5>

## News
* ```2024.10.14``` 🔥 We release the [Demo page](https://hanbo-cheng.github.io/DAWN/).
* ```2024.10.18``` 🔥 We release the paper [DAWN](https://arxiv.org/abs/2410.13726).
* ```2024.10.21``` 🔥 We update the Chinese introduction [知乎](https://zhuanlan.zhihu.com/p/2253009511).
## TODO list:
- [ ]  release the inference code (before December)
- [ ]  release the pretrained model of **128*128** (before December)
- [ ]  release the pretrained model of **256*256** (before December)
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

Thank you to the authors of [Diffused Heads](https://github.com/MStypulkowski/diffused-heads) for assisting us in reproducing their work! We also extend our gratitude to the authors of [MRAA](https://github.com/snap-research/articulated-animation), [LFDM](https://github.com/snap-research/articulated-animation) and [ACTOR](https://github.com/Mathux/ACTOR) for their contributions to the open-source community. Lastly, we thank our mentors and co-authors for their continuous support in our research work!


# 🌅 DAWN：Dynamic Frame Avatar with Non-autoregressive Diffusion Framework for Talking Head Video Generation

[![arXiv](https://img.shields.io/badge/Arxiv-2410.13726-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.13726)
[![Demo Page](https://img.shields.io/badge/Demo_Page-blue)](https://hanbo-cheng.github.io/DAWN/)
[![zhihu](https://img.shields.io/badge/知乎-0079FF.svg?logo=zhihu&logoColor=white)](https://zhuanlan.zhihu.com/p/2253009511)
 <a href='https://huggingface.co/Hanbo-Cheng/DAWN'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
<p align="center">

<img src="structure_img\ifferent-styles-at-higher-resolution.gif" width=600>
</p>


😊 请给我们一个star⭐支持我们的持续更新 😊
## 新闻
* ```2024.10.14``` 🔥 我们发布了 [DEMO](https://hanbo-cheng.github.io/DAWN/)。
* ```2024.10.18``` 🔥 我们发布了论文 [DAWN](https://arxiv.org/abs/2410.13726)。
* ```2024.10.21``` 🔥 我们更新了中文介绍 [知乎](https://zhuanlan.zhihu.com/p/2253009511)。
* ```2024.11.7``` 🔥🔥 我们在 [hugging face](https://huggingface.co/Hanbo-Cheng/DAWN) 上发布了预训练模型。
* ```2024.11.9``` 🔥🔥🔥 我们发布了推理代码。我们诚挚邀请您体验我们的模型。😊
*  ```2025.2.16``` 🔥🔥🔥 我们优化了统一推理代码。现在您可以仅用一个脚本运行测试流程。🚀

## 待办事项列表：
- [x]  发布推理代码
- [x]  发布 **128*128** 的预训练模型
- [x]  发布 **256*256** 的预训练模型 
- [x] 发布统一测试代码
- [ ]  进行中 ...

## 设备要求

使用我们针对VRAM优化的 [代码](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test_local_opt.py)，生成的视频最大长度与GPU VRAM的大小 **成线性关系**。更大的VRAM可以生成更长的视频。
- 要生成 **128*128** 视频，我们建议使用 **12GB** 或更多VRAM的GPU。这至少可以生成大约 **400帧** 的视频。
- 要生成 **256*256** 视频，我们建议使用 **24GB** 或更多VRAM的GPU。这至少可以生成大约 **200帧** 的视频。

PS：尽管优化的 [代码](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test_local_opt.py) 可以提高VRAM利用率，但由于局部注意力的优化尚不完整，目前牺牲了推理速度。我们正在积极解决这个问题，如果您有更好的解决方案，欢迎您提交PR。如果您希望实现更快的推理速度，可以使用 [未优化的代码](DM_3/modules/video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test.py)，但这将增加VRAM使用（O(n²) 空间复杂度）。

## 方法论
### DAWN的整体结构：
<p align="center">
<img src="structure_img\pipeline.png" width=600 alt="framework"/>
</p>

## 环境
我们强烈建议在Linux平台上尝试DAWN。在Windows上运行可能会产生一些需要手动删除的垃圾文件，并且需要额外的努力来部署3DDFA库（我们的 `extract_init_states` 文件夹） [评论](https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173)。

1. 设置conda环境
```
conda create -n DAWN python=3.8
conda activate DAWN
pip install -r requirements.txt
```

2. 按照 [readme](extract_init_states/readme.md) 和 [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) 设置3DDFA环境。

## 推理

由于我们的模型 **仅在HDTF数据集上训练**，并且参数较少，为了确保最佳的驱动效果，请尽量提供以下示例：
- 尽量使用标准人像照片，避免佩戴帽子或大型头饰
- 确保背景与主体之间有清晰的边界
- 确保面部在图像中占据主要位置。

推理准备：
1. 从 [hugging face](https://huggingface.co/Hanbo-Cheng/DAWN) 下载预训练检查点。创建 `./pretrain_models` 目录并将检查点文件放入其中。请从 [facebook/hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft/tree/main) 下载Hubert模型。
   
2. 运行推理脚本： 
   ```
   python unified_video_generator.py  \
      --audio_path your/audio/path  \
      --image_path your/image/path  \
      --output_path output/path \
      --cache_path cache/path 
   ```

***在其他数据集上的推理：***
通过在每次推理时指定 `VideoGenerator` 类的 `audio_path`、`image_path` 和 `output_path`，并修改 `unified_video_generator.py` 中第310-312行和393-394行的 `directory_name` 和 `output_video_path` 的内容，您可以控制保存图像和视频的命名逻辑，从而在任何数据集上进行测试。



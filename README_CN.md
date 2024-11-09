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
* ```2024.10.14``` 🔥 我们发布了[Demo page](https://hanbo-cheng.github.io/DAWN/)。
* ```2024.10.18``` 🔥 我们发布了论文[DAWN](https://arxiv.org/abs/2410.13726)。
* ```2024.10.21``` 🔥 我们更新了中文介绍。
* ```2024.11.7```  🔥 我们在[hugging face](https://huggingface.co/Hanbo-Cheng/DAWN)上发布了预训练模型。
* ```2024.11.9``` 🔥 我们上传了推理代码。我们诚挚邀请您体验我们的模型。 😊
  
## TODO list:
- [x] 发布推理代码
- [x] 发布128*128的预训练模型
- [x] 发布256*256的预训练模型
- [ ] 发布HDTF数据集的测试代码
- [ ] 进行中...
  
## 设备要求
使用我们针对显存优化的代码，生成的视频长度与显存的大小**线性相关**。更大的显存可以生成更长的视频。 
- 为了生成**128 * 128**视频，建议使用至少具有**12GB** 显存的GPU。这可以生成大约**400帧**的视频。 
- 为了生成**256 * 256**视频，建议使用至少具有**24GB** 显存的GPU。这可以生成大约**200帧**的视频。

PS: 尽管经过优化的[code](DM_3\modules\video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test_local_opt.py)可以提高显存的利用率，但目前牺牲了推理速度，因为局部注意力的优化不完整。我们正在积极解决这个问题，如果您有更好的解决方案，欢迎您的PR。如果您希望获得更快的推理速度，可以使用[未优化的代码](DM_3\modules\video_flow_diffusion_multiGPU_v0_crema_plus_faceemb_ca_multi_test.py)，但这会增加显存的使用（O(n²)空间复杂度）。

## 方法论
### DAWN的整体结构：
<p align="center">
<img src="structure_img\pipeline.png" width=600 alt="framework"/>
</p>

## 环境
我们强烈建议在Linux平台上尝试DAWN。在Windows上运行可能会产生一些需要手动删除的垃圾文件，并且在Windows平台上运行需要额外步骤部署3DDFA环境（我们的extract_init_states）[comment](https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173)。

1. 设置conda环境
```
conda create -n DAWN python=3.8
conda activate DAWN
pip install -r requirements.txt
conda create -n 3DDFA python=3.8
conda activate 3DDFA
pip install -r requirements_3ddfa.txt
```

1. 遵循(extract_init_states\readme.md) 和 [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)设置3DDFA环境。

## 推理
1. 从[hugging face](https://huggingface.co/Hanbo-Cheng/DAWN)下载预训练检查。创建`./pretrain_models`目录并将检查点文件放入其中。
2. 修改`run_ood_test\run_DM_v0_df_test_128_both_pose_blink.sh` 或者 `run_ood_test\run_DM_v0_df_test_256_1.sh`中的路径。这包括了`image_path`, driving `audio_path` and `cache_path`。`run_ood_test\run_DM_v0_df_test_128_both_pose_blink.sh`用于实现 128 * 128 视频的推理， `run_ood_test\run_DM_v0_df_test_256_1.sh` 用于实现 256 * 256视频的推理。

3. 使用 `bash xxxx.sh` 根据你的需求运行上述的脚本。

这段代码在公司内部服务器和我的Windows 11个人电脑上进行了测试。由于设备之间的差异，可能会出现一些细微问题。如果遇到问题，请随时提出问题或提交PR，我们乐意提供帮助！

**用于数据集测试**：如果您希望在数据集上测试DAWN的性能，我们建议将我们的代码进行修改，并为每个步骤批量处理数据（包括提取初始状态、音频嵌入、PBNet推理和A2V-FDM推理）。反复加载模型将会大大降低您的测试效率。我们还计划在未来发布HDTF数据集的推理代码。


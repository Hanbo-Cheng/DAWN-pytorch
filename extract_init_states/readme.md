# README

The `extract_init_state` is mainly from [3DDFA_v2](https://github.com/cleardusk/3DDFA_V2) with minor revision. We remove the `Sim3DR` in original repo.
We add or revise the file of `extract_init_states\demo_pose_extract_2d_lmk_img.py`, `extract_init_states\utils\pose.py`.

## Linux
Linux user can follow the installation process on [3DDFA_v2](https://github.com/cleardusk/3DDFA_V2)

## Win
For Windows user, be aware to these tips:
1. Installing gcc
2. In `extract_init_states\FaceBoxes\utils\build.py`, you need comment line 47
3. Revise the `extract_init_states\FaceBoxes\utils\nms\cpu_nms.pyx` following [comment](https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173).
4. Run the command in sh script line by line manually



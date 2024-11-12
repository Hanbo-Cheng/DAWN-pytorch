

source /home4/intern/hbcheng2/.bashrc


test_name=ood_test_1006 # $(date +"%Y-%m-%d_%H-%M-%S")
time_tag=tmp #$(date +"%Y-%m-%d_%H-%M-%S")
audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip7.wav
image_path=your/path/DAWN-pytorch/ood_data/ood_select_3/test4.jpeg
cache_path=your/path/DAWN-pytorch/ood_data_3/$time_tag
audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip7.npy

conda activate 3DDFA
cd /train20/intern/permanent/hbcheng2/AIGC_related/3DDFA_V2-master
python /train20/intern/permanent/hbcheng2/AIGC_related/3DDFA_V2-master/demo_pose_extract_2d_lmk_img.py \
    --input $image_path \
    --output $cache_path

# conda activate LFDM_chb
# cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
#     --src_audio_path $audio_path \
#     --save_path $audio_emb_path

conda activate LFDM_chb
cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
    --audio_path  $audio_emb_path \
    --init_pose_blink $cache_path \
    --output $cache_path

cd your/path/DAWN-pytorch
# source /home4/intern/hbcheng2/.bashrc

# echo 'finish extracting init state'
python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
    --source_img_path $image_path \
    --init_state_path $cache_path \
    --drive_blink_path $cache_path/dri_blink.npy \
    --drive_pose_path $cache_path/dri_pose.npy \
    --audio_emb_path $audio_emb_path \
    --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
    --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip1.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip1.npy


# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'

# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip2.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip2.npy

# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path



# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'

# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip3.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip3.npy


# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'
# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip4.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip4.npy


# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'
# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip5.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip5.npy


# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'

# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path

# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip6.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip6.npy



# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'
# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path



# audio_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_clip_vocal_origin/Taylor-Swift-You-Belong-With-Me-vocal_clip0.wav
# # image_path=your/path/DAWN-pytorch/ood_data/ood_select/images/draw_female_test1.png
# # cache_path=your/path/DAWN-pytorch/ood_data_3/$test_name
# audio_emb_path=your/path/DAWN-pytorch/ood_data/ood_select/audio_embedding_vocal/Taylor-Swift-You-Belong-With-Me-vocal_clip0.npy



# # conda activate LFDM_chb
# # cd /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main
# # python /train20/intern/permanent/hbcheng2/AIGC_related/GeneFace-main/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
# #     --src_audio_path $audio_path \
# #     --save_path $audio_emb_path


# cd /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/evaluate/tvae_eval_signal.py \
#     --audio_path  $audio_emb_path \
#     --init_pose_blink $cache_path \
#     --output $cache_path

# cd your/path/DAWN-pytorch
# # source /home4/intern/hbcheng2/.bashrc
# # conda activate LFDM_a40
# # echo 'finish extracting init state'

# python your/path/DAWN-pytorch/DM_1/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
#     --source_img_path $image_path \
#     --init_state_path $cache_path \
#     --drive_blink_path $cache_path/dri_blink.npy \
#     --drive_pose_path $cache_path/dri_pose.npy \
#     --audio_emb_path $audio_emb_path \
#     --save_path /train20/intern/permanent/hbcheng2/data/ood_test_3/$test_name \
#     --src_audio_path $audio_path




test_name=ood_test_1009 # $(date +"%Y-%m-%d_%H-%M-%S")
time_tag=tmp1009 # $(date +"%Y-%m-%d_%H-%M-%S")
audio_path=WRA_MarcoRubio_000.wav 
image_path=real_female_1.jpeg
cache_path=cache/$time_tag
audio_emb_path=cache/target_audio.npy
video_output_path=cache/

source activate
conda activate 3DDFA
cd extract_init_states
python demo_pose_extract_2d_lmk_img.py \
    --input ../$image_path \
    --output ../$cache_path

cd ..
conda activate DAWN

python ./hubert_extract/data_gen/process_lrs3/process_audio_hubert_interpolate_demo.py \
    --src_audio_path $audio_path \
    --save_path $audio_emb_path

python ./PBnet/src/evaluate/tvae_eval_single.py \
    --audio_path  $audio_emb_path \
    --init_pose_blink $cache_path \
    --output $cache_path \
    --ckpt_pose ./pretrain_models/pbnet_seperate/pose/checkpoint_40000.pth.tar \
    --ckpt_blink ./pretrain_models/pbnet_seperate/blink/checkpoint_95000.pth.tar 



python ./DM_3/test_demo/test_VIDEO_hdtf_df_wpose_face_cond_init_ca_newae_ood_256_2.py --gpu 0  \
    --source_img_path $image_path \
    --init_state_path $cache_path \
    --drive_blink_path $cache_path/dri_blink.npy \
    --drive_pose_path $cache_path/dri_pose.npy \
    --audio_emb_path $audio_emb_path \
    --save_path $video_output_path/$test_name \
    --src_audio_path $audio_path


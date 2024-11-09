source /home4/intern/lmlin2/.bashrc
conda activate actor
# crema rc delta pose
export CUDA_VISIBLE_DEVICES="0"
# python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/train/train_cvae.py\
#      --num_frames 40\
#      --lambda_kl 1\
#      --lambda_ssim 1\
#      --lambda_freq 1\
#      --modelname cvae_transformer_ssim_kl_freq\
#      --dataset hdtf\
#      --num_epochs 10000\
#      --folder exps_delta_pose/HDTF_nf40_kl1_ssim1_freq_128_w5_1w_6

python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/train/train_cvae_ganloss_ann_eye.py\
     --num_frames 200\
     --eye True\
     --lr 0.0004 \
     --batch_size 40\
     --lambda_kl 0.004\
     --lambda_reg 0.0005\
     --lambda_rc 1\
     --ff_size 128\
     --max_distance 128\
     --num_buckets 128\
     --num_layers 2\
     --audio_latent_dim 256\
     --snapshot 10000\
     --modelname cvae_transformerreemb8_rc_kl_reg\
     --dataset hdtf\
     --num_epochs 100000\
     --folder exps_delta_pose_rope_eye/HDTF_b40_200_eye_kl4e3_lr4e-4_reg5e-4_rope16_3 #  > output.log &

# nohup python /train20/intern/permanent/hbcheng2/AIGC_related/ACTOR-master/src/train/train_cvae_ganloss_first3.py\
#      --num_frames 40\
#      --batch_size 20\
#      --lambda_kl 1\
#      --lambda_rc 1\
#      --num_layers 4\
#      --modelname cvae_transformerold_kl_ssim\
#      --dataset hdtf\
#      --num_epochs 30000\
#      --folder exps_delta_pose_f3/HDTF_l2_nf40_kl1_ssim_norm_w5_1w_b20_first_3 > output.log &

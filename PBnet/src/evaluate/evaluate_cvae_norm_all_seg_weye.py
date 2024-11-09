import sys
sys.path.append('your_path/PBnet')

from src.parser.evaluation import parser
from src.datasets.datasets_crema_pos_eye_fast_all import CREMA
from src.datasets.datasets_hdtf_pos_chunk_norm_2_all import HDTF
from src.evaluate.tvae_eval_norm_seg import evaluate



def main():
    parameters, folder, checkpointname, epoch, niter = parser()
    
    # data path

    dataset_name = parameters["dataset"]
    if dataset_name == 'crema':
        # data path
        data_dir = "/work1/cv2/pcxia/diffusion_v1/diffused-heads-colab-main/datasets/images"
        # model and dataset
        dataset = CREMA(data_dir=data_dir,
                        max_num_frames=parameters["num_frames"],
                        mode = 'test')
        dataset.update_parameters(parameters)
    elif dataset_name == 'hdtf':
        data_dir = "/yrfs2/cv2/pcxia/audiovisual/hdtf/images_25hz"
        dataset = HDTF(data_dir=data_dir,
                        max_num_frames=1e8,
                        mode = 'test')
        # dataset.update_parameters(parameters)
        parameters["audio_dim"] = 1024
        parameters["pos_dim"] = 6
    else:
        dataset = None
        print('Dataset can not be found!!')

    evaluate(parameters, dataset, folder, checkpointname, epoch, niter)



if __name__ == '__main__':
    main()

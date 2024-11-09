import sys
sys.path.append('/train20/intern/permanent/lmlin2/ReferenceCode/ACTOR-master')

from src.parser.evaluation import parser
from src.datasets.datasets_crema_pos import CREMA
from src.datasets.datasets_hdtf_onlyeye_fast import HDTF
from src.evaluate.tvae_eval_onlyeye_all_seg import evaluate


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
        dataset.update_parameters(parameters)
        # parameters["audio_dim"] = 1024
        # parameters["pos_dim"] = 6
    else:
        dataset = None
        print('Dataset can not be found!!')

    evaluate(parameters, dataset, folder, checkpointname, epoch, niter)



if __name__ == '__main__':
    main()

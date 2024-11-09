from src.models.get_model import LOSSES, MODELTYPES, ARCHINAMES


def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--modelname", default='cvae_transformer_rc_kl', help="Choice of the model, should be like cvae_transformer_rc_rcxyz_kl")
    group.add_argument("--latent_dim", default=256, type=int, help="dimensionality of the latent space")
    group.add_argument("--lambda_kl", default=1.0, type=float, help="weight of the kl divergence loss")
    group.add_argument("--lambda_rcw", default=1.0, type=float, help="weight of the rc divergence loss with weight")
    group.add_argument("--lambda_rc", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_ssim", default=1.0, type=float, help="weight of the ssim divergence loss")
    group.add_argument("--lambda_reg", default=0.1, type=float, help="weight of the reg loss")
    # group.add_argument("--lambda_var", default=-0.1, type=float, help="weight of the var divergence loss")

    group.add_argument("--num_layers", default=2, type=int, help="Number of layers for GRU and transformer")
    group.add_argument("--ff_size", default=128, type=int, help="Size of feedforward for transformer")
    group.add_argument("--max_distance", default=128, type=int, help="")
    group.add_argument("--num_buckets", default=128, type=int, help="")
    group.add_argument("--audio_latent_dim", default=256, type=int, help="Size of audio latent for transformer")
    group.add_argument("--first3", default=False, help="Dim of pose, 3 or 6")
    group.add_argument("--eye", default=False, help="eye information")
    group.add_argument("--activation", default="gelu", help="Activation for function for the transformer layers")
    group.add_argument("--dropout", default=0.1, type=float, help="Activation for function for the transformer layers")

    # # Ablations
    # group.add_argument("--ablation", choices=[None, "average_encoder", "zandtime", "time_encoding", "concat_bias"],
    #                    help="Ablations for the transformer architechture")


def parse_modelname(modelname):
    modeltype, archiname, *losses = modelname.split("_")

    if modeltype not in MODELTYPES:
        raise NotImplementedError("This type of model is not implemented.")
    if archiname not in ARCHINAMES:
        raise NotImplementedError("This architechture is not implemented.")

    if len(losses) == 0:
        raise NotImplementedError("You have to specify at least one loss function.")

    for loss in losses:
        if loss not in LOSSES:
            raise NotImplementedError("This loss is not implemented.")

    return modeltype, archiname, losses

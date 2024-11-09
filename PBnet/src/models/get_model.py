import importlib

# JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "kl", "rcw", "ssim", "var", 'reg']  # not used: "hp", "mmd", "vel", "velxyz"

MODELTYPES = ["cvae"]  # not used: "cae"
ARCHINAMES = ["fc", "gru", "transformer","transformerreemb5", "transformerreemb6", "transformerreemb7", "transformerreemb8","transformermel", "transgru", "grutrans", "autotrans"]


def get_model(parameters):
    modeltype = parameters["modeltype"]
    archiname = parameters["archiname"]

    archi_module = importlib.import_module(f'.architectures.{archiname}', package="src.models")
    Encoder = archi_module.__getattribute__(f"Encoder_{archiname.upper()}")
    Decoder = archi_module.__getattribute__(f"Decoder_{archiname.upper()}")

    model_module = importlib.import_module(f'.modeltype.{modeltype}', package="src.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")

    encoder = Encoder(**parameters)
    decoder = Decoder(**parameters)
    
    # parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return Model(encoder, decoder, **parameters).to(parameters["device"])

"""
read wav file and load model tools
"""
import torchaudio
from .dprnn import DprnnModel

# path_list: a list of wav path
# return_type: return numpy array or tensor


def read_wav(path_list, return_type="np", rate=8000):
    """
    read wav file
    """
    assert return_type in ["np", "tensor"], "return_type must in ['np', 'tensor']."

    if return_type == "tensor":
        assert len(path_list) == 1, "If return_type is tensor, length of path_list must be 1."
        try:
            batch, _ = torchaudio.sox_effects.apply_effects_file(path_list[0], effects=[["rate", f"{rate}"]])
        except Exception as e:
            raise e
        return batch

    wav_list = []
    for path in path_list:
        try:
            batch, _ = torchaudio.sox_effects.apply_effects_file(path, effects=[["rate", f"{rate}"]])
        except Exception as e:
            raise e
        wav_list.append(batch)

    return [b.numpy() for b in wav_list]


# model_path : the folder contain model
# device:  model device

def load_model(model_path, device):
    """
    load model
    """
    run_opts = {"device": device}
    model = DprnnModel.from_hparams(source=model_path, savedir=model_path, hparams_file='dprnn.yaml', run_opts=run_opts)
    return model

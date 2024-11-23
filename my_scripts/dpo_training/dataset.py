from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from torch.nn.utils.rnn import pad_sequence


class DPODataset(XTTSDataset):
    def __init__(self, config, samples, tokenizer, sample_rate, is_eval=False):
        super().__init__(config, samples, tokenizer, sample_rate, is_eval=False)

    def __getitem__(self, index):
        res = super().__getitem__(index)
        res |= {
            'mel_cond_w': res['sample']['mel_cond_w'],
            'mel_cond_l': res['sample']['mel_cond_l'],
        }
        return res

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        batch['mel_cond_w'] = pad_sequence(batch['mel_cond_w'], batch_first=True, padding_value=1025)#[:, :-1]
        batch['mel_cond_l'] = pad_sequence(batch['mel_cond_l'], batch_first=True, padding_value=1025)#[:, :-1]
        del batch['sample']
        # del batch['audio_codes']
        return batch
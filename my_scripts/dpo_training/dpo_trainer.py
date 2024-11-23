from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts
from TTS.utils.io import load_fsspec
from my_scripts.dpo_training.dataset import DPODataset

@dataclass
class DPOArgs(GPTArgs):
    beta: float = 1.0



class DPOTrainer(BaseTTS):
    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.ref_xtts = Xtts(self.config)
        self.xtts = Xtts(self.config)

        # create the tokenizer with the target vocabulary
        self.ref_xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.ref_xtts.init_models()
        self.xtts.init_models()

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

        # set mel stats
        if self.args.mel_norm_file:
            self.ref_xtts.mel_stats = load_fsspec(self.args.mel_norm_file)
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device("cpu"))
            # deal with coqui Trainer exported model
            if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
                print("Coqui Trainer checkpoint detected! Converting it!")
                gpt_checkpoint = gpt_checkpoint["model"]
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if "gpt." in key:
                        new_key = key.replace("gpt.", "")
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                    "text_embedding.weight" in gpt_checkpoint
                    and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                        self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print(">> GPT weights restored from:", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
            self,
            text_inputs,
            text_lengths,
            wav_lengths,
            cond_mels,
            cond_idxs,
            cond_lens,
            mel_cond_w,
            mel_cond_l,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        with torch.inference_mode():
            _, ref_mel_loss_w, (ref_logits_w, ref_target_w) = self.ref_xtts.gpt(
                text_inputs,
                text_lengths,
                mel_cond_w,
                wav_lengths,
                cond_mels=cond_mels,
                cond_idxs=cond_idxs,
                cond_lens=cond_lens,
                reduction='none',
            )
            _, ref_mel_loss_l, (ref_logits_l, ref_target_l) = self.ref_xtts.gpt(
                text_inputs,
                text_lengths,
                mel_cond_l,
                wav_lengths,
                cond_mels=cond_mels,
                cond_idxs=cond_idxs,
                cond_lens=cond_lens,
                reduction='none',
            )
        _, mel_loss_w, (logits_w, target_w) = self.xtts.gpt(
            text_inputs,
            text_lengths,
            mel_cond_w,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            reduction='none',
        )
        _, mel_loss_l, (logits_l, target_l) = self.xtts.gpt(
            text_inputs,
            text_lengths,
            mel_cond_l,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            reduction='none',
        )

        mel_loss_w = -mel_loss_w
        mel_loss_l = -mel_loss_l
        ref_mel_loss_w = -ref_mel_loss_w
        ref_mel_loss_l = -ref_mel_loss_l

        loss_w = mel_loss_w - ref_mel_loss_w
        loss_l = mel_loss_l - ref_mel_loss_l

        # return (ref_logits_w, ref_logits_l), (logits_w, logits_l), (loss_w, loss_l)
        return {
            'ref_logits_w': ref_logits_w,
            'ref_logits_l': ref_logits_l,
            'ref_mel_loss_w': ref_mel_loss_w,
            'ref_mel_loss_l': ref_mel_loss_l,
            'logits_w': logits_w,
            'logits_l': logits_l,
            'mel_loss_w': mel_loss_w,
            'mel_loss_l': mel_loss_l,
            'loss_w': loss_w,
            'loss_l': loss_l,
        }

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.config,
                    s_info["speaker_wav"],
                    s_info["language"],
                    gpt_cond_len=3,
                )["wav"]
                test_audios["{}-audio".format(idx)] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
            self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)

        batch["audio_codes"] = codes
        # delete useless batch tensors
        del batch["padded_text"]
        del batch["wav"]
        del batch["conditioning"]
        return batch

    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]
        mel_cond_w = batch['mel_cond_w']
        mel_cond_l = batch['mel_cond_l']

        output = self.forward(
            text_inputs, text_lengths, wav_lengths, cond_mels, cond_idxs, cond_lens, mel_cond_w, mel_cond_l
        )

        loss = torch.log(torch.sigmoid(self.args.beta*(output['loss_w'] - output['loss_l']))).mean()

        loss_dict['loss_ref_mel_w'] = output['ref_mel_loss_w'].mean()
        loss_dict['loss_ref_mel_l'] = output['ref_mel_loss_l'].mean()
        loss_dict['loss_mel_w'] = output['mel_loss_w'].mean()
        loss_dict['loss_mel_l'] = output['mel_loss_l'].mean()
        loss_dict['loss_w'] = output['loss_w'].mean()
        loss_dict['loss_l'] = output['loss_l'].mean()
        loss_dict['loss'] = loss
        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        trainer.model.eval()  # the whole model to eval
        # put gpt model in training mode
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            trainer.model.module.xtts.gpt.train()
        else:
            trainer.model.xtts.gpt.train()

    @torch.no_grad()
    def inference(
            self,
            x,
            aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
            self,
            config: Coqpit,
            assets: Dict,
            is_eval: bool,
            samples: Union[List[Dict], List[List]],
            verbose: bool,
            num_gpus: int,
            rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = DPODataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
            self,
            config,
            checkpoint_path,
            eval=False,
            strict=True,
            cache_storage="/tmp/tts_cache",
            target_protocol="s3",
            target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""

        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return DPOTrainer(config)

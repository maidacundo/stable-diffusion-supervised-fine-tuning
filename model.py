from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional

def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    device: str = "cuda:0",
    load_from_safetensor=True,
):
    if load_from_safetensor:

        print('loading VAE...')
        vae = AutoencoderKL.from_single_file(
            pretrained_vae_name_or_path,
        )
        print('loading model...')
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            pretrained_model_name_or_path,
            vae=vae,
            safety_checker=None,
        )
        tokenizer = pipe.tokenizer 
        text_encoder = pipe.text_encoder
        unet = pipe.unet

    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        )

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        )
    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
    )

def get_ddim_scheduler(pretrained_model_name_or_path):
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    return scheduler



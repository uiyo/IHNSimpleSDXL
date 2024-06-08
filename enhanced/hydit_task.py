import os
import tarfile
import numpy as np
import torch
import gc
import ldm_patched.modules.model_management as model_management

from io import BytesIO
from PIL import Image
from pathlib import Path
from hydit.constants import SAMPLER_FACTORY
from hydit.config import get_args
from hydit.inference import End2End
from modules.config import path_t2i, add_ratio
from modules.model_loader import load_file_from_url
from modules.launch_util import is_installed
from diffusers import HunyuanDiTPipeline
from transformers import T5EncoderModel

default_aspect_ratio = add_ratio('1024*1024')
available_aspect_ratios = [
        '768*1280', '960*1280', '1024*1024', 
        '1280*768', '1280*960', '1280*1280',
    ]
available_aspect_ratios = [add_ratio(x) for x in available_aspect_ratios]

SAMPLERS = list(SAMPLER_FACTORY.keys())
default_sampler = SAMPLERS[0]
infer_mode = "fa" if is_installed("flash-attn") else "torch"
new_args = ["--use_fp16", "--lang", "zh", "--load-key", "distill", "--infer-mode", infer_mode]
hydit_args = get_args(new_args)
gen = None

hydit_models_root = Path(os.path.join(path_t2i, "t2i"))
hydit_text_encoder = None
hydit_pipe = None

def init_load_model():
    global hydit_models_root, hydit_pipe

    check_files_exist = lambda ph, fs: all(os.path.exists(os.path.join(ph, f)) for f in fs)

    files = ["clip_text_encoder/pytorch_model.bin", "model/pytorch_model_module.pt", "mt5/pytorch_model.bin", "sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"]
    if not hydit_models_root.exists() or not check_files_exist(hydit_models_root, files):
        hydit_models_root.mkdir(parents=True, exist_ok=True)
        downloading_hydit_model(hydit_models_root)
    
    if 'hydit_text_encoder' not in globals():
        globals()['hydit_text_encoder'] = None
    if hydit_text_encoder is None:
        hydit_text_encoder = T5EncoderModel.from_pretrained(
            hydit_models_root,
            subfolder="text_encoder_2",
            load_in_8bit=True,
            device_map="auto",
        )
    if 'hydit_pipe' not in globals():
        globals()['hydit_pipe'] = None
    if hydit_pipe is None:
        hydit_pipe = HunyuanDiTPipeline.from_pretrained(
            hydit_models_root, 
            text_encoder_2=text_encoder_2,
            transformer=None,
            vae=None,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
    print("[HyDiT] Initialized the HyDit environment and loaded model files.")

def unload_free_model():
    global hydit_pipe

    if 'hydit_pipe' in globals():
        del hydit_pipe
    model_management.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    print("[HyDiT] Freed the GPU Ram occupyed by the HyDit.")

def get_scheduler_name(sampler):
    params = SAMPLER_FACTORY[sampler]
    return params["scheduler"], params["name"]


def inferencer(
    prompt,
    negative_prompt,
    seed,
    cfg_scale,
    infer_steps,
    width, height,
    sampler,
):
    global hydit_models_root

    if 'hydit_pipe' not in globals():
        globals()['hydit_pipe'] = None
    if hydit_pipe is None:
        init_load_model()

    seed = seed & 0xFFFFFFFF
    enhanced_prompt = None
    params = SAMPLER_FACTORY[sampler]
    print(f'[HyDiT] Ready to HyDiT Task:\n    prompt={prompt}\n    negative_prompt={negative_prompt}\n    seed={seed}\n    cfg_scale={cfg_scale}\n    steps={infer_steps}\n    width,height={width},{height}\n    scheduler={params["scheduler"]}\n    sampler={params["name"]}')

    
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask = hydit_pipe.encode_prompt(prompt)
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
        ) = hydit_pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            max_sequence_length=256,
            text_encoder_index=1,
        )
    unload_free_model()

    if sampler=='ddpm':
        scheduler = DDPMScheduler.from_config(params['kwargs'])
    elif sampler=='ddim':
        scheduler = DDIMScheduler.from_config(params['kwargs'])
    elif sampler=='dpmms':
        scheduler = DPMSolverMultistepScheduler.from_config(params['kwargs'])
    else:
        raise ValueError(f'The sampler:{sampler} is not in SAMPLER_FACTORY')

    device = model_management.get_torch_device()
    pipe = HunyuanDiTPipeline.from_pretrained(
        hydit_models_root,
        text_encoder=None,
        text_encoder_2=None,
        scheduler=params["scheduler"],
        torch_dtype=torch.float16,
        )
    pipe.scheduler = scheduler
    pipe.to(device)

    image = pipe(
        height=height,
        width=width,
        num_inference_steps=infer_steps,
        guidance_scale=cfg_scale,
        num_images_per_prompt=1,
        generator=torch.Generator(device=device).manual_seed(seed)
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_2=prompt_embeds_2,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_2=negative_prompt_embeds_2,
        prompt_attention_mask=prompt_attention_mask,
        prompt_attention_mask_2=prompt_attention_mask_2,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
    ).images[0]

    return [np.array(image)]

def downloading_hydit_model(path_root):
    load_file_from_url(
        url='https://huggingface.co/metercai/backup/resolve/main/hydit_models.tar.gz',
        model_dir=path_root,
        file_name='hydit_models.tar.gz'
    )
    downfile = os.path.join(path_root, 'hydit_models.tar.gz')
    with tarfile.open(downfile, 'r:gz') as tarf:
        tarf.extractall(path_root)
    os.remove(downfile)
    pass

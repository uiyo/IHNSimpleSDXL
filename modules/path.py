import os
import json
import args_manager
import modules.flags
import modules.sdxl_styles

from modules.model_loader import load_file_from_url
from modules.util import get_files_from_folder

from fooocus_extras.controlnet_preprocess_model.PyramidCanny import PyramidCanny
from fooocus_extras.controlnet_preprocess_model.CPDS import CPDS
from fooocus_extras.controlnet_preprocess_model.ZeoDepth import ZoeDetector
from fooocus_extras.controlnet_preprocess_model.OpenPose import OpenPose
from fooocus_extras.controlnet_preprocess_model.ReColor import ReColor
from fooocus_extras.controlnet_preprocess_model.Sketch import Sketch
from fooocus_extras.controlnet_preprocess_model.Revision import Revision
from fooocus_extras.controlnet_preprocess_model.TileBlur import TileBlur
from fooocus_extras.controlnet_preprocess_model.TileBlurAnime import TileBlurAnime

config_path = "user_path_config.txt"
config_dict = {}

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict = json.load(json_file)
except Exception as e:
    print('Load path config failed')
    print(e)

preset = args_manager.args.preset

if isinstance(preset, str):
    preset = os.path.abspath(f'./presets/{preset}.json')
    try:
        if os.path.exists(preset):
            with open(preset, "r", encoding="utf-8") as json_file:
                preset = json.load(json_file)
    except Exception as e:
        print('Load preset config failed')
        print(e)

preset = preset if isinstance(preset, dict) else None

if preset is not None:
    config_dict.update(preset)




def get_dir_or_set_default(key, default_value):
    global config_dict
    v = config_dict.get(key, None)
    if isinstance(v, str) and os.path.exists(v) and os.path.isdir(v):
        return v
    else:
        root_dir = os.path.abspath(os.path.dirname(__file__))
        dp = os.path.abspath(os.path.join(root_dir, default_value))
        os.makedirs(dp, exist_ok=True)
        config_dict[key] = dp
        return dp


modelfile_path = get_dir_or_set_default('modelfile_path', '../models/checkpoints/')
lorafile_path = get_dir_or_set_default('lorafile_path', '../models/loras/')
embeddings_path = get_dir_or_set_default('embeddings_path', '../models/embeddings/')
vae_approx_path = get_dir_or_set_default('vae_approx_path', '../models/vae_approx/')
upscale_models_path = get_dir_or_set_default('upscale_models_path', '../models/upscale_models/')
inpaint_models_path = get_dir_or_set_default('inpaint_models_path', '../models/inpaint/')
controlnet_models_dir = get_dir_or_set_default('controlnet_models_path', '../models/controlnet/')
clip_vision_models_path = get_dir_or_set_default('clip_vision_models_path', '../models/clip_vision/')
fooocus_expansion_path = get_dir_or_set_default('fooocus_expansion_path',
                                                '../models/prompt_expansion/fooocus_expansion')
temp_outputs_path = get_dir_or_set_default('temp_outputs_path', '../outputs/')


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False):
    global config_dict
    if key not in config_dict:
        config_dict[key] = default_value
        return default_value

    v = config_dict.get(key, None)
    if not disable_empty_as_none:
        if v is None or v == '':
            v = 'None'
    if validator(v):
        return v
    else:
        config_dict[key] = default_value
        return default_value


default_base_model_name = get_config_item_or_set_default(
    key='default_model',
    default_value='sd_xl_base_1.0_0.9vae.safetensors',
    validator=lambda x: isinstance(x, str)
)
default_refiner_model_name = get_config_item_or_set_default(
    key='default_refiner',
    default_value='sd_xl_refiner_1.0_0.9vae.safetensors',
    validator=lambda x: isinstance(x, str)
)
default_lora_name = get_config_item_or_set_default(
    key='default_lora',
    default_value='sd_xl_offset_example-lora_1.0.safetensors',
    validator=lambda x: isinstance(x, str)
)
default_lora_weight = get_config_item_or_set_default(
    key='default_lora_weight',
    default_value=0.5,
    validator=lambda x: isinstance(x, float)
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, float)
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value='dpmpp_2m_sde_gpu',
    validator=lambda x: x in modules.flags.sampler_list
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value='karras',
    validator=lambda x: x in modules.flags.scheduler_list
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=['Fooocus V2', 'Default (Slightly Cinematic)'],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x)
)
default_negative_prompt = get_config_item_or_set_default(
    key='default_negative_prompt',
    default_value='low quality, bad hands, bad eyes, cropped, missing fingers, extra digit',
    validator=lambda x: isinstance(x, str)
)
default_positive_prompt = get_config_item_or_set_default(
    key='default_positive_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={
        'sd_xl_base_1.0_0.9vae.safetensors':
            'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors',
        'sd_xl_refiner_1.0_0.9vae.safetensors':
            'https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors',
    },
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={
        'sd_xl_offset_example-lora_1.0.safetensors':
            'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors'
    },
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items())
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1152*896',
    validator=lambda x: x.replace('*', '×') in modules.sdxl_styles.aspect_ratios
).replace('*', '×')

if preset is None:
    # Do not overwrite user config if preset is applied.
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, indent=4)

os.makedirs(temp_outputs_path, exist_ok=True)

model_filenames = []
lora_filenames = []


def get_model_filenames(folder_path, name_filter=None):
    return get_files_from_folder(folder_path, ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch'], name_filter)


def update_all_model_names():
    global model_filenames, lora_filenames
    model_filenames = get_model_filenames(modelfile_path)
    lora_filenames = get_model_filenames(lorafile_path)
    return


def downloading_inpaint_models(v):
    assert v in ['v1', 'v2.5']

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=inpaint_models_path,
        file_name='fooocus_inpaint_head.pth'
    )

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=inpaint_models_path,
            file_name='inpaint.fooocus.patch'
        )
        return os.path.join(inpaint_models_path, 'fooocus_inpaint_head.pth'), os.path.join(inpaint_models_path,
                                                                                           'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=inpaint_models_path,
            file_name='inpaint_v25.fooocus.patch'
        )
        return os.path.join(inpaint_models_path, 'fooocus_inpaint_head.pth'), os.path.join(inpaint_models_path,
                                                                                           'inpaint_v25.fooocus.patch')


def GET_PATH(m):
    dir = m['dir']
    file_name = m['file_name']
    if dir is None or file_name is None:
        return None
    else:
        return os.path.join(dir, file_name)


CONTROLNET_MODELS = [
    # ControlnetModels
    {
        'id': 0,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-depth-rank128.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'control-lora-depth-rank128.safetensors',
        'loader': 'ControlNet',
        'condition': "depth",
        'preprocess': False,  # is preprocess model?
        'default': True,  # default model?
        'path': GET_PATH,
    },
    {
        'id': 1,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'control-lora-canny-rank128.safetensors',
        'loader': 'ControlNet',
        'condition': "canny",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 2,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose_256lora.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'thibaud_xl_openpose_256lora.safetensors',
        'loader': 'ControlNet',
        'condition': "pose",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 3,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'fooocus_xl_cpds_128.safetensors',
        'loader': 'ControlNet',
        'condition': "cpds",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 4,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'control-lora-recolor-rank128.safetensors',
        'loader': 'ControlNet',
        'condition': "recolor",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 5,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'control-lora-sketch-rank128-metadata.safetensors',
        'loader': 'ControlNet',
        'condition': "sketch",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 6,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'clip_vision_g.safetensors',
        'loader': 'ControlNet',
        'condition': "revision",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 7,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'kohya_controllllite_xl_blur.safetensors',
        'loader': 'ControlNet',
        'condition': "tile_blur",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 8,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime.safetensors'
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'kohya_controllllite_xl_blur_anime.safetensors',
        'loader': 'ControlNet',
        'condition': "tile_blur_anime",
        'preprocess': False,
        'default': True,
        'path': GET_PATH,
    },
    # preprocessing
    {
        'id': 9,
        'url': None,
        'dir': None,
        'file_name': 'ControlNetCannyPreprocess__pyramidCanny',
        'loader': PyramidCanny,
        'condition': "canny",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 10,
        'url': None,
        'dir': None,
        'file_name': 'ControlNetCPDSPreprocess__cpds',
        'loader': CPDS,
        'condition': "cpds",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 11,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt',
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'ZoeD_M12_N.pt',
        'loader': ZoeDetector,
        'condition': "depth",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 12,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth',
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'body_pose_model.pth',
        'loader': OpenPose,
        'condition': "pose",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 13,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth',
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'hand_pose_model.pth',
        'loader': OpenPose,
        'condition': "pose",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 14,
        'url': {
            'select': 0,
            'providers': [
                'https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth',
            ]
        },
        'dir': controlnet_models_dir,
        'file_name': 'facenet.pth',
        'loader': OpenPose,
        'condition': "pose",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 15,
        'url': None,
        'dir': controlnet_models_dir,
        'file_name': 'ControlNetReColorPreprocess_reColor',
        'loader': ReColor,
        'condition': "recolor",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 16,
        'url': None,
        'dir': controlnet_models_dir,
        'file_name': 'ControlNetSketchPreprocess_sketch',
        'loader': Sketch,
        'condition': "sketch",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 17,
        'url': None,
        'dir': controlnet_models_dir,
        'file_name': 'ControlNetRevisionPreprocess_revision',
        'loader': Revision,
        'condition': "revision",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 18,
        'url': None,
        'dir': controlnet_models_dir,
        'file_name': 'ControlNetTileBlurPreprocess_tileBlur',
        'loader': TileBlur,
        'condition': "tile_blur",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },
    {
        'id': 19,
        'url': None,
        'dir': controlnet_models_dir,
        'file_name': 'ControlNetTileBlurAnimePreprocess_tileBlurAnime',
        'loader': TileBlurAnime,
        'condition': "tile_blur_anime",
        'preprocess': True,
        'default': True,
        'path': GET_PATH,
    },

]


def downloading_controlnet_models(condition):
    def download(m):
        url = m['url']
        if m['default']:
            if url is not None:
                select = url['select']
                providers = url['providers']
                if not select < len(providers):
                    raise ValueError(f"Invalid url select: {select} for {providers}")
                load_file_from_url(
                    url=providers[select],
                    model_dir=m['dir'],
                    file_name=m['file_name']
                )
        return m

    if condition not in set([model['condition'] for model in CONTROLNET_MODELS]):
        raise ValueError(f"Invalid condition: {condition}")
    models = [download(m) for m in CONTROLNET_MODELS if m['condition'] == condition]
    controlnet_models = [m for m in models if m['loader'] == 'ControlNet']
    preprocess_models = [m for m in models if m['loader'] != 'ControlNet']
    assert 1 == len(controlnet_models)
    assert 1 <= len(preprocess_models)
    return models


def downloading_ip_adapters():
    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=clip_vision_models_path,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(clip_vision_models_path, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=controlnet_models_dir,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(controlnet_models_dir, 'fooocus_ip_negative.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
        model_dir=controlnet_models_dir,
        file_name='ip-adapter-plus_sdxl_vit-h.bin'
    )
    results += [os.path.join(controlnet_models_dir, 'ip-adapter-plus_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=upscale_models_path,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(upscale_models_path, 'fooocus_upscaler_s409985e5.bin')


update_all_model_names()
import os
import json
import math
import numbers

import args_manager
import tempfile
import shared
import modules.flags
import modules.sdxl_styles
import enhanced.all_parameters as ads

from modules.model_loader import load_file_from_url
from modules.extra_utils import makedirs_with_log, get_files_from_folder, try_eval_env_var
from modules.flags import OutputFormat, Performance, MetadataScheme



def get_config_path(key, default_value):
    env = os.getenv(key)
    if env is not None and isinstance(env, str):
        print(f"Environment: {key} = {env}")
        return env
    else:
        return os.path.abspath(default_value)

wildcards_max_bfs_depth = 64
config_path = get_config_path('config_path', "./config.txt") if args_manager.args.config is None else os.path.abspath(os.path.join(args_manager.args.config, "config.txt"))
config_example_path = get_config_path('config_example_path', "config_modification_tutorial.txt")
config_dict = {}
always_save_keys = []
visited_keys = []

try:
    with open(os.path.abspath(f'./presets/default.json'), "r", encoding="utf-8") as json_file:
        config_dict.update(json.load(json_file))
except Exception as e:
    print(f'Load default preset failed.')
    print(e)

try:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_dict.update(json.load(json_file))
        always_save_keys = list(config_dict.keys())
        for key in always_save_keys:
            if key.startswith('default_') and key[8:] in ads.default:
                ads.default[key[8:]] = config_dict[key]
        print(f'Load config data from {config_path}.')
except Exception as e:
    print(f'Failed to load config file "{config_path}" . The reason is: {str(e)}')
    print('Please make sure that:')
    print(f'1. The file "{config_path}" is a valid text file, and you have access to read it.')
    print('2. Use "\\\\" instead of "\\" when describing paths.')
    print('3. There is no "," before the last "}".')
    print('4. All key/value formats are correct.')


def try_load_deprecated_user_path_config():
    global config_dict

    if not os.path.exists('user_path_config.txt'):
        return

    try:
        deprecated_config_dict = json.load(open('user_path_config.txt', "r", encoding="utf-8"))

        def replace_config(old_key, new_key):
            if old_key in deprecated_config_dict:
                config_dict[new_key] = deprecated_config_dict[old_key]
                del deprecated_config_dict[old_key]

        replace_config('modelfile_path', 'path_checkpoints')
        replace_config('lorafile_path', 'path_loras')
        replace_config('embeddings_path', 'path_embeddings')
        replace_config('vae_approx_path', 'path_vae_approx')
        replace_config('upscale_models_path', 'path_upscale_models')
        replace_config('inpaint_models_path', 'path_inpaint')
        replace_config('controlnet_models_path', 'path_controlnet')
        replace_config('clip_vision_models_path', 'path_clip_vision')
        replace_config('fooocus_expansion_path', 'path_fooocus_expansion')
        replace_config('temp_outputs_path', 'path_outputs')

        if deprecated_config_dict.get("default_model", None) == 'juggernautXL_version6Rundiffusion.safetensors':
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully in silence. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return

        if input("Newer models and configs are available. "
                 "Download and update files? [Y/n]:") in ['n', 'N', 'No', 'no', 'NO']:
            config_dict.update(deprecated_config_dict)
            print('Loading using deprecated old models and deprecated old configs.')
            return
        else:
            os.replace('user_path_config.txt', 'user_path_config-deprecated.txt')
            print('Config updated successfully by user. '
                  'A backup of previous config is written to "user_path_config-deprecated.txt".')
            return
    except Exception as e:
        print('Processing deprecated config failed')
        print(e)
    return


try_load_deprecated_user_path_config()

preset = args_manager.args.preset
theme = args_manager.args.theme

def get_presets():
    preset_folder = 'presets'
    presets = ['initial']
    if not os.path.exists(preset_folder):
        print('No presets found.')
        return presets

    return presets + [f[:f.index('.json')] for f in os.listdir(preset_folder) if f.endswith('.json')]


def try_get_preset_content(preset):
    if isinstance(preset, str):
        preset_path = os.path.abspath(f'./presets/{preset}.json')
        try:
            if os.path.exists(preset_path):
                with open(preset_path, "r", encoding="utf-8") as json_file:
                    json_content = json.load(json_file)
                    print(f'Loaded preset: {preset_path}')
                    return json_content
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f'Load preset [{preset_path}] failed')
            print(e)
    return {}

available_presets = get_presets()
preset = args_manager.args.preset
config_dict.update(try_get_preset_content(preset))

def get_path_output() -> str:
    """
    Checking output path argument and overriding default path.
    """
    global config_dict
    path_output = get_dir_or_set_default('path_outputs', '../outputs/', make_directory=True)
    if args_manager.args.output_path:
        print(f'Overriding config value path_outputs with {args_manager.args.output_path}')
        config_dict['path_outputs'] = path_output = args_manager.args.output_path
    return path_output


def get_dir_or_set_default(key, default_value, as_array=False, make_directory=False):
    global config_dict, visited_keys, always_save_keys

    if key not in visited_keys:
        visited_keys.append(key)

    if key not in always_save_keys:
        always_save_keys.append(key)

    v = os.getenv(key)
    if v is not None:
        print(f"Environment: {key} = {v}")
        config_dict[key] = v
    else:
        v = config_dict.get(key, None)

    if isinstance(v, str):
        if make_directory:
            makedirs_with_log(v)
        if os.path.exists(v) and os.path.isdir(v):
            return v if not as_array else [v]
    elif isinstance(v, list):
        if make_directory:
            for d in v:
                makedirs_with_log(d)
        if all([os.path.exists(d) and os.path.isdir(d) for d in v]):
            return v

    if v is not None:
        print(f'Failed to load config key: {json.dumps({key:v})} is invalid or does not exist; will use {json.dumps({key:default_value})} instead.')
    if isinstance(default_value, list):
        dp = []
        for path in default_value:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            dp.append(abs_path)
            os.makedirs(abs_path, exist_ok=True)
    else:
        dp = os.path.abspath(os.path.join(os.path.dirname(__file__), default_value))
        os.makedirs(dp, exist_ok=True)
        if as_array:
            dp = [dp]
    config_dict[key] = dp
    return dp


paths_checkpoints = get_dir_or_set_default('path_checkpoints', ['../models/checkpoints/'], True)
paths_loras = get_dir_or_set_default('path_loras', ['../models/loras/'], True)
path_embeddings = get_dir_or_set_default('path_embeddings', '../models/embeddings/')
path_vae_approx = get_dir_or_set_default('path_vae_approx', '../models/vae_approx/')
path_vae = get_dir_or_set_default('path_vae', '../models/vae/')
path_upscale_models = get_dir_or_set_default('path_upscale_models', '../models/upscale_models/')
path_inpaint = get_dir_or_set_default('path_inpaint', '../models/inpaint/')
path_controlnet = get_dir_or_set_default('path_controlnet', '../models/controlnet/')
path_clip_vision = get_dir_or_set_default('path_clip_vision', '../models/clip_vision/')
path_fooocus_expansion = get_dir_or_set_default('path_fooocus_expansion', '../models/prompt_expansion/fooocus_expansion')
path_llms = get_dir_or_set_default('path_llms','../models/llms/')
path_wildcards = get_dir_or_set_default('path_wildcards', '../wildcards/')
path_safety_checker = get_dir_or_set_default('path_safety_checker', '../models/safety_checker/')
path_outputs = get_path_output()
path_models_root = get_dir_or_set_default('path_models_root', '../models/')
path_unet = get_dir_or_set_default('path_unet', '../models/unet')
path_rembg = get_dir_or_set_default('path_rembg', '../models/rembg')
path_layer_model = get_dir_or_set_default('path_layer_model', '../models/layer_model')


def get_config_item_or_set_default(key, default_value, validator, disable_empty_as_none=False, expected_type=None):
    global config_dict, visited_keys

    if key not in visited_keys:
        visited_keys.append(key)
    
    v = os.getenv(key)
    if v is not None:
        v = try_eval_env_var(v, expected_type)
        print(f"Environment: {key} = {v}")
        config_dict[key] = v

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
        if v is not None:
            print(f'Failed to load config key: {json.dumps({key:v})} is invalid; will use {json.dumps({key:default_value})} instead.')
        config_dict[key] = default_value
        return default_value

def init_temp_path(path: str | None, default_path: str) -> str:
    if args_manager.args.temp_path:
        path = args_manager.args.temp_path
    
    if path != '' and path != default_path:
        try:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            os.makedirs(path, exist_ok=True)
            print(f'Using temp path {path}')
            return path
        except Exception as e:
            print(f'Could not create temp path {path}. Reason: {e}')
            print(f'Using default temp path {default_path} instead.')

    os.makedirs(default_path, exist_ok=True)
    return default_path


default_loras = get_config_item_or_set_default(
    key='default_loras',
    default_value=[
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ],
        [
            True,
            "None",
            1.0
        ]
    ],
    validator=lambda x: isinstance(x, list) and all(
        len(y) == 3 and isinstance(y[0], bool) and isinstance(y[1], str) and isinstance(y[2], numbers.Number)
        or len(y) == 2 and isinstance(y[0], str) and isinstance(y[1], numbers.Number)
        for y in x)
)
default_loras = [(y[0], y[1], y[2]) if len(y) == 3 else (True, y[0], y[1]) for y in default_loras]
default_max_lora_number = get_config_item_or_set_default(
    key='default_max_lora_number',
    default_value=len(default_loras) if isinstance(default_loras, list) and len(default_loras) > 0 else ads.default['max_lora_number'],
    validator=lambda x: isinstance(x, int) and x >= 1
)

ads.init_all_params_index(default_max_lora_number, args_manager.args.disable_metadata)

default_temp_path = os.path.join(tempfile.gettempdir(), 'fooocus')

temp_path = init_temp_path(get_config_item_or_set_default(
    key='temp_path',
    default_value=default_temp_path,
    validator=lambda x: isinstance(x, str),
    expected_type=str
), default_temp_path)
temp_path_cleanup_on_launch = get_config_item_or_set_default(
    key='temp_path_cleanup_on_launch',
    default_value=True,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_base_model_name = default_model = get_config_item_or_set_default(
    key='default_model',
    default_value='model.safetensors',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
previous_default_models = get_config_item_or_set_default(
    key='previous_default_models',
    default_value=[],
    validator=lambda x: isinstance(x, list) and all(isinstance(k, str) for k in x),
    expected_type=list
)
default_refiner_model_name = default_refiner = get_config_item_or_set_default(
    key='default_refiner',
    default_value='None',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_refiner_switch = get_config_item_or_set_default(
    key='default_refiner_switch',
    default_value=0.8,
    validator=lambda x: isinstance(x, numbers.Number) and 0 <= x <= 1,
    expected_type=numbers.Number
)
default_loras_min_weight = get_config_item_or_set_default(
    key='default_loras_min_weight',
    default_value=ads.default['loras_min_weight'],
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_loras_max_weight = get_config_item_or_set_default(
    key='default_loras_max_weight',
    default_value=ads.default['loras_max_weight'],
    validator=lambda x: isinstance(x, numbers.Number) and -10 <= x <= 10,
    expected_type=numbers.Number
)
default_cfg_scale = get_config_item_or_set_default(
    key='default_cfg_scale',
    default_value=7.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sample_sharpness = get_config_item_or_set_default(
    key='default_sample_sharpness',
    default_value=2.0,
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_sampler = get_config_item_or_set_default(
    key='default_sampler',
    default_value=ads.default['sampler_name'],
    validator=lambda x: x in modules.flags.sampler_list,
    expected_type=str
)
default_scheduler = get_config_item_or_set_default(
    key='default_scheduler',
    default_value=ads.default['scheduler_name'],
    validator=lambda x: x in modules.flags.scheduler_list,
    expected_type=str
)
default_vae = get_config_item_or_set_default(
    key='default_vae',
    default_value=modules.flags.default_vae,
    validator=lambda x: isinstance(x, str),
    expected_type=str
)
default_styles = get_config_item_or_set_default(
    key='default_styles',
    default_value=[
        "Fooocus V2",
        "Fooocus Enhance",
        "Fooocus Sharp"
    ],
    validator=lambda x: isinstance(x, list) and all(y in modules.sdxl_styles.legal_style_names for y in x),
    expected_type=list
)
default_prompt_negative = get_config_item_or_set_default(
    key='default_prompt_negative',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_prompt = get_config_item_or_set_default(
    key='default_prompt',
    default_value='',
    validator=lambda x: isinstance(x, str),
    disable_empty_as_none=True,
    expected_type=str
)
default_performance = get_config_item_or_set_default(
    key='default_performance',
    default_value=Performance.SPEED.value,
    validator=lambda x: x in Performance.list(),
    expected_type=str
)
default_advanced_checkbox = get_config_item_or_set_default(
    key='default_advanced_checkbox',
    default_value=ads.default['advanced_checkbox'],
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_max_image_number = get_config_item_or_set_default(
    key='default_max_image_number',
    default_value=ads.default['max_image_number'],
    validator=lambda x: isinstance(x, int) and x >= 1,
    expected_type=int
)
default_output_format = get_config_item_or_set_default(
    key='default_output_format',
    default_value=ads.default['output_format'],
    validator=lambda x: x in OutputFormat.list(),
    expected_type=str
)
default_image_number = get_config_item_or_set_default(
    key='default_image_number',
    default_value=ads.default['image_number'],
    validator=lambda x: isinstance(x, int) and 1 <= x <= default_max_image_number,
    expected_type=int
)
checkpoint_downloads = get_config_item_or_set_default(
    key='checkpoint_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
lora_downloads = get_config_item_or_set_default(
    key='lora_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
embeddings_downloads = get_config_item_or_set_default(
    key='embeddings_downloads',
    default_value={},
    validator=lambda x: isinstance(x, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in x.items()),
    expected_type=dict
)
available_aspect_ratios = get_config_item_or_set_default(
    key='available_aspect_ratios',
    default_value=modules.flags.sdxl_aspect_ratios,
    validator=lambda x: isinstance(x, list) and all('*' in v for v in x) and len(x) > 1,
    expected_type=list
)
default_aspect_ratio = get_config_item_or_set_default(
    key='default_aspect_ratio',
    default_value='1152*896' if '1152*896' in available_aspect_ratios else available_aspect_ratios[0],
    validator=lambda x: x in available_aspect_ratios,
    expected_type=str
)
default_inpaint_engine_version = get_config_item_or_set_default(
    key='default_inpaint_engine_version',
    default_value=ads.default['inpaint_engine'],
    validator=lambda x: x in modules.flags.inpaint_engine_versions,
    expected_type=str
)
default_cfg_tsnr = get_config_item_or_set_default(
    key='default_cfg_tsnr',
    default_value=ads.default['adaptive_cfg'],
    validator=lambda x: isinstance(x, numbers.Number),
    expected_type=numbers.Number
)
default_clip_skip = get_config_item_or_set_default(
    key='default_clip_skip',
    default_value=2,
    validator=lambda x: isinstance(x, int) and 1 <= x <= modules.flags.clip_skip_max,
    expected_type=int
)
default_overwrite_step = get_config_item_or_set_default(
    key='default_overwrite_step',
    default_value=ads.default['overwrite_step'],
    validator=lambda x: isinstance(x, int),
    expected_type=int
)
default_overwrite_switch = get_config_item_or_set_default(
    key='default_overwrite_switch',
    default_value=ads.default['overwrite_switch'],
    validator=lambda x: isinstance(x, int),
    expected_type=int
)

default_inpaint_mask_upload_checkbox = get_config_item_or_set_default(
    key='default_inpaint_mask_upload_checkbox',
    default_value=ads.default['inpaint_mask_upload_checkbox'],
    validator=lambda x: isinstance(x, bool)
)

example_inpaint_prompts = get_config_item_or_set_default(
    key='example_inpaint_prompts',
    default_value=[
        'highly detailed face', 'detailed girl face', 'detailed man face', 'detailed hand', 'beautiful eyes'
    ],
    validator=lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x),
    expected_type=list
)
default_black_out_nsfw = get_config_item_or_set_default(
    key='default_black_out_nsfw',
    default_value=False,
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_save_metadata_to_images = get_config_item_or_set_default(
    key='default_save_metadata_to_images',
    default_value=ads.default['save_metadata_to_images'],
    validator=lambda x: isinstance(x, bool),
    expected_type=bool
)
default_metadata_scheme = get_config_item_or_set_default(
    key='default_metadata_scheme',
    default_value=ads.default['metadata_scheme'],
    validator=lambda x: x in [y[1] for y in modules.flags.metadata_scheme if y[1] == x],
    expected_type=str
)
metadata_created_by = get_config_item_or_set_default(
    key='metadata_created_by',
    default_value='',
    validator=lambda x: isinstance(x, str),
    expected_type=str
)

example_inpaint_prompts = [[x] for x in example_inpaint_prompts]

default_inpaint_mask_model = get_config_item_or_set_default(
    key='default_inpaint_mask_model',
    default_value='isnet-general-use',
    validator=lambda x: x in modules.flags.inpaint_mask_models
)

default_inpaint_mask_cloth_category = get_config_item_or_set_default(
    key='default_inpaint_mask_cloth_category',
    default_value='full',
    validator=lambda x: x in modules.flags.inpaint_mask_cloth_category
)

default_inpaint_mask_sam_model = get_config_item_or_set_default(
    key='default_inpaint_mask_sam_model',
    default_value='sam_vit_b_01ec64',
    validator=lambda x: x in modules.flags.inpaint_mask_sam_model
)

default_translation_methods = get_config_item_or_set_default(
    key='default_translation_methods',
    default_value=ads.default['translation_methods'],
    validator=lambda x: x in modules.flags.translation_methods
)

default_backfill_prompt = get_config_item_or_set_default(
    key='default_backfill_prompt',
    default_value=ads.default['backfill_prompt'],
    validator=lambda x: isinstance(x, bool)
)

default_backend = get_config_item_or_set_default(
    key='default_backend',
    default_value=ads.default['backend'],
    validator=lambda x: x in modules.flags.backend_engines
)

default_comfyd_active_checkbox = get_config_item_or_set_default(
    key='default_comfyd_active_checkbox',
    default_value=ads.default['comfyd_active_checkbox'],
    validator=lambda x: isinstance(x, bool)
)

config_dict["default_loras"] = default_loras = default_loras[:default_max_lora_number] + [[True, 'None', 1.0] for _ in range(default_max_lora_number - len(default_loras))]

# mapping config to meta parameter 
possible_preset_keys = {
    "default_model": "base_model",
    "default_refiner": "refiner_model",
    "default_refiner_switch": "refiner_switch",
    "previous_default_models": "previous_default_models",
    "default_loras_min_weight": "default_loras_min_weight",
    "default_loras_max_weight": "default_loras_max_weight",
    "default_loras": "<processed>",
    "default_cfg_scale": "guidance_scale",
    "default_sample_sharpness": "sharpness",
    "default_cfg_tsnr": "adaptive_cfg",
    "default_clip_skip": "clip_skip",
    "default_sampler": "sampler",
    "default_scheduler": "scheduler",
    "default_overwrite_step": "steps",
    "default_performance": "performance",
    "default_image_number": "image_number",
    "default_prompt": "prompt",
    "default_prompt_negative": "negative_prompt",
    "default_styles": "styles",
    "default_aspect_ratio": "resolution",
    "default_cfg_tsnr": "cfg_tsnr",
    "default_overwrite_step": "overwrite_step",
    "default_overwrite_switch": "overwrite_switch",
    "default_save_metadata_to_images": "default_save_metadata_to_images",
    "checkpoint_downloads": "checkpoint_downloads",
    "embeddings_downloads": "embeddings_downloads",
    "lora_downloads": "lora_downloads",
    "default_vae": "vae"
}

REWRITE_PRESET = False

if REWRITE_PRESET and isinstance(args_manager.args.preset, str):
    save_path = 'presets/' + args_manager.args.preset + '.json'
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in possible_preset_keys}, json_file, indent=4)
    print(f'Preset saved to {save_path}. Exiting ...')
    exit(0)


def add_ratio(x):
    a, b = x.replace('*', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    c, d = a // g, b // g
    if (a, b) == (576, 1344):
        c, d = 9, 21
    elif (a, b) == (1344, 576):
        c, d = 21, 9
    elif (a, b) == (768, 1280):
        c, d = 9, 15
    elif (a, b) == (1280, 768):
        c, d = 15, 9
    return f'{a}×{b} <span style="color: grey;"> \U00002223 {c}:{d}</span>'


default_aspect_ratio = add_ratio(default_aspect_ratio)
available_aspect_ratios_labels = [add_ratio(x) for x in available_aspect_ratios]

#sd3_default_aspect_ratio = '16:9'
#sd3_available_aspect_ratios = ['21:9', '16:9', '3:2', '5:4', '1:1', '2:3', '4:5', '9:16', '9:21']
sd3_default_aspect_ratio = add_ratio('1024*1024')
sd3_available_aspect_ratios = [
        '576*1344', '768*1152', '896*1152', '768*1280', '960*1280',  
        '1024*1024', '1024*1280', '1280*1280', '1280*1024',
        '1280*960', '1280*768', '1152*896', '1152*768', '1344*576'
    ]
sd3_available_aspect_ratios = [add_ratio(x) for x in sd3_available_aspect_ratios]


# Only write config in the first launch.
if not os.path.exists(config_path):
    with open(config_path, "w", encoding="utf-8") as json_file:
        json.dump({k: config_dict[k] for k in always_save_keys}, json_file, indent=4)


# Always write tutorials.
with open(config_example_path, "w", encoding="utf-8") as json_file:
    cpa = config_path.replace("\\", "\\\\")
    json_file.write(f'You can modify your "{cpa}" using the below keys, formats, and examples.\n'
                    f'Do not modify this file. Modifications in this file will not take effect.\n'
                    f'This file is a tutorial and example. Please edit "{cpa}" to really change any settings.\n'
                    + 'Remember to split the paths with "\\\\" rather than "\\", '
                      'and there is no "," before the last "}". \n\n\n')
    json.dump({k: config_dict[k] for k in visited_keys}, json_file, indent=4)

config_comfy_path = os.path.join(shared.root, 'comfy/extra_model_paths.yaml')
config_comfy_formatted_text = '''
comfyui:
     checkpoints: {checkpoints} 
     clip_vision: {clip_vision}
     controlnet: {controlnet}
     embeddings: {embeddings}
     loras: {loras}
     upscale_models: {upscale_models}
     unet: {unet}
     rembg: {rembg}
     layer_model: {layer_model}
     '''

paths2str = lambda p: '\n'.join(p[:-1]) + ('' if not p else p[-1])
config_comfy_text = config_comfy_formatted_text.format(checkpoints=paths2str(paths_checkpoints), clip_vision=path_clip_vision, controlnet=path_controlnet, embeddings=path_embeddings, loras=paths2str(paths_loras), upscale_models=path_upscale_models, unet=path_unet, rembg=path_rembg, layer_model=path_layer_model)
with open(config_comfy_path, "w", encoding="utf-8") as comfy_file:
    comfy_file.write(config_comfy_text)


model_filenames = []
lora_filenames = []
vae_filenames = []
wildcard_filenames = []


def get_model_filenames(folder_paths, extensions=None, name_filter=None):
    if extensions is None:
        extensions = ['.pth', '.ckpt', '.bin', '.safetensors', '.fooocus.patch']
    files = []

    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]
    for folder in folder_paths:
        files += get_files_from_folder(folder, extensions, name_filter)

    return files


def update_files():
    global model_filenames, lora_filenames, vae_filenames, wildcard_filenames, available_presets
    model_filenames = get_model_filenames(paths_checkpoints)
    lora_filenames = get_model_filenames(paths_loras)
    vae_filenames = get_model_filenames(path_vae)
    wildcard_filenames = get_files_from_folder(path_wildcards, ['.txt'])
    available_presets = get_presets()
    return


def downloading_inpaint_models(v):
    assert v in modules.flags.inpaint_engine_versions

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')
    patch_file = None

    if v == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')

    if v == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')

    if v == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')

    return head_file, patch_file


def downloading_sdxl_lcm_lora():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/sdxl_lcm_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.EXTREME_SPEED.value
    )
    return modules.flags.PerformanceLoRA.EXTREME_SPEED.value


def downloading_sdxl_lightning_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_lightning_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.LIGHTNING.value
    )
    return modules.flags.PerformanceLoRA.LIGHTNING.value


def downloading_sdxl_hyper_sd_lora():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/sdxl_hyper_sd_4step_lora.safetensors',
        model_dir=paths_loras[0],
        file_name=modules.flags.PerformanceLoRA.HYPER_SD.value
    )
    return modules.flags.PerformanceLoRA.HYPER_SD.value


def downloading_controlnet_canny():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/control-lora-canny-rank128.safetensors',
        model_dir=path_controlnet,
        file_name='control-lora-canny-rank128.safetensors'
    )
    return os.path.join(path_controlnet, 'control-lora-canny-rank128.safetensors')


def downloading_controlnet_cpds():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_xl_cpds_128.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_xl_cpds_128.safetensors'
    )
    return os.path.join(path_controlnet, 'fooocus_xl_cpds_128.safetensors')


def downloading_ip_adapters(v):
    assert v in ['ip', 'face']

    results = []

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/clip_vision_vit_h.safetensors',
        model_dir=path_clip_vision,
        file_name='clip_vision_vit_h.safetensors'
    )
    results += [os.path.join(path_clip_vision, 'clip_vision_vit_h.safetensors')]

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_ip_negative.safetensors',
        model_dir=path_controlnet,
        file_name='fooocus_ip_negative.safetensors'
    )
    results += [os.path.join(path_controlnet, 'fooocus_ip_negative.safetensors')]

    if v == 'ip':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus_sdxl_vit-h.bin')]

    if v == 'face':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/ip-adapter-plus-face_sdxl_vit-h.bin',
            model_dir=path_controlnet,
            file_name='ip-adapter-plus-face_sdxl_vit-h.bin'
        )
        results += [os.path.join(path_controlnet, 'ip-adapter-plus-face_sdxl_vit-h.bin')]

    return results


def downloading_upscale_model():
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin',
        model_dir=path_upscale_models,
        file_name='fooocus_upscaler_s409985e5.bin'
    )
    return os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')

def downloading_dwposeprocess_model():
    results = []
    # yolox_l.onnx
    load_file_from_url(
        url='https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx',
        model_dir=path_controlnet,
        file_name='yolox_l.onnx'
    )
    results += [os.path.join(path_controlnet, 'yolox_l.onnx')]
    # dw-ll_ucoco_384.onnx
    load_file_from_url(
        url='https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx',
        model_dir=path_controlnet,
        file_name='dw-ll_ucoco_384.onnx'
    )
    results += [os.path.join(path_controlnet, 'dw-ll_ucoco_384.onnx')]
    return results


def downloading_openposeprocess_model():
    results = []
    # openpose lora
    load_file_from_url(
        url='https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors',
        model_dir=path_controlnet,
        file_name='thibaud_xl_openpose_256lora.safetensors'
    )
    results += [os.path.join(path_controlnet, 'thibaud_xl_openpose_256lora.safetensors')]
    # openpose body
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth',
        model_dir=path_controlnet,
        file_name='body_pose_model.pth'
    )
    results += [os.path.join(path_controlnet, 'body_pose_model.pth')]

    # openpose hand
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth',
        model_dir=path_controlnet,
        file_name='hand_pose_model.pth'
    )
    results += [os.path.join(path_controlnet, 'hand_pose_model.pth')]
    
    # openpose face
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth',
        model_dir=path_controlnet,
        file_name='facenet.pth'
    )
    results += [os.path.join(path_controlnet, 'facenet.pth')]
    return results
def downloading_safety_checker_model():
    load_file_from_url(
        url='https://huggingface.co/mashb1t/misc/resolve/main/stable-diffusion-safety-checker.bin',
        model_dir=path_safety_checker,
        file_name='stable-diffusion-safety-checker.bin'
    )
    return os.path.join(path_safety_checker, 'stable-diffusion-safety-checker.bin')


def downloading_superprompter_model():
    path_superprompter = os.path.join(path_llms, "superprompt-v1")
    load_file_from_url(
        url='https://huggingface.co/roborovski/superprompt-v1/resolve/main/model.safetensors',
        model_dir=path_superprompter,
        file_name='model.safetensors'
    )
    return os.path.join(path_superprompter, 'model.safetensors')

def downloading_sd3_medium_model():
    load_file_from_url(
        url='https://huggingface.co/metercai/SimpleSDXL2/resolve/main/sd3_medium_incl_clips.safetensors',
        model_dir=paths_checkpoints[0],
        file_name='sd3_medium_incl_clips.safetensors'
    )
    return os.path.join(paths_checkpoints[0], 'sd3_medium_incl_clips.safetensors')

update_files()
from enhanced.simpleai import simpleai_config, refresh_models_info 
simpleai_config.paths_checkpoints = paths_checkpoints
simpleai_config.paths_loras = paths_loras
simpleai_config.path_embeddings = path_embeddings

refresh_models_info()




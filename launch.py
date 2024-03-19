import os
import sys
import ssl

#print('[System PATH] ' + str(sys.path))
print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["translators_default_region"] = "China"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context


import platform
import fooocus_version
import enhanced.version as version

from build_launcher import build_launcher, is_win32_standalone_build, python_embeded_path
from modules.launch_util import is_installed, run, python, run_pip, requirements_met
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    torch_command += ' -i https://pypi.tuna.tsinghua.edu.cn/simple '
    target_path_win = os.path.join(python_embeded_path, 'Lib/site-packages')
    if is_win32_standalone_build:
        torch_command += f' -t {target_path_win}'

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")
    print(f'{version.get_branch()} version: {version.get_simplesdxl_ver()}')

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
       
        if is_win32_standalone_build:
            import modules.launch_util as launch_util
            if len(launch_util.met_diff.keys())>0:
                for p in launch_util.met_diff.keys():
                    print(f'Uninstall {p}.{launch_util.met_diff[p]} ...')
                    run(f'"{python}" -m pip uninstall -y {p}=={launch_util.met_diff[p]}')
            run_pip(f"install -r \"{requirements_file}\" -t {target_path_win}", "requirements")
        else:
            run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v3.1.safetensors',
     'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
]

def ini_args():
    from args_manager import args
    return args


def is_ipynb():
    return True if 'ipykernel' in sys.modules and hasattr(sys, '_jupyter_kernel') else False

prepare_environment()
#build_launcher()
args = ini_args()


if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

import enhanced.token_did as token_did
token_did.init_local_did(f'SimpleSDXL_User')

import enhanced.location as location 
location.init_location()

if '--location' in sys.argv:
        location.location = args.location

if location.location !='CN':
    if '--language' not in sys.argv:
        args.language='default'

import socket
if '--listen' not in sys.argv:
    if is_ipynb():
        args.listen = '0.0.0.0'
    else:
        args.listen = socket.gethostbyname(socket.gethostname())
if '--port' not in sys.argv:
    args.port = 6067

from modules import config

def download_models():
    from modules.model_loader import load_file_from_url
    from modules import config

    os.environ["U2NET_HOME"] = config.path_inpaint
    os.environ["HUF_MIRROR"] = 'hf-mirror.com'

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return

    if not args.always_download_new_model:
        if not os.path.exists(os.path.join(config.paths_checkpoints[0], config.default_base_model_name)):
            for alternative_model_name in config.previous_default_models:
                if os.path.exists(os.path.join(config.paths_checkpoints[0], alternative_model_name)):
                    print(f'You do not have [{config.default_base_model_name}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    config.checkpoint_downloads = {}
                    config.default_base_model_name = alternative_model_name
                    break

    for file_name, url in config.checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_checkpoints[0], file_name=file_name)
    for file_name, url in config.embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in config.lora_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_loras[0], file_name=file_name)

    return


download_models()


from webui import *

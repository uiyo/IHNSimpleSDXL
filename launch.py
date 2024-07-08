import os
import ssl
import sys
import json
import importlib
import packaging.version
from pathlib import Path

print('[System PATH] ' + str(sys.path))
# print('[System ARGV] ' + str(sys.argv))

root = Path(__file__).resolve().parent

sys.path.append(str(root / 'modules'))
sys.path.append(str(root))
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["translators_default_region"] = "China"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform

from build_launcher import build_launcher, is_win32_standalone_build, python_embeded_path
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content, git_clone, index_url, target_path_install, met_diff

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = True

target_path_win = os.path.join(python_embeded_path, 'Lib/site-packages')

def check_base_environment():
#    sys.path.append(os.path.join(root, "comfy"))
    sys.path.append(os.path.join(root, "hydit"))

    import fooocus_version
    import comfy.comfy_version as comfy_version
    import enhanced.version as version

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")
    print(f"Comfy version: {comfy_version.version}")
    print(f'{version.get_branch()} version: {version.get_simplesdxl_ver()}')

    base_pkg = "simpleai_base"
    ver_required = "0.3.15"
    REINSTALL_BASE = False
    base_file = {
        "Windows": f'enhanced/libs/simpleai_base-{ver_required}-cp310-none-win_amd64.whl',
        "Linux": f'enhanced/libs/simpleai_base-{ver_required}-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
        }
    #index_url = "https://pypi.org/simple"
    if not is_installed(base_pkg):
        run(f'"{python}" -m pip install {base_file[platform.system()]}', f'Install {base_pkg} {ver_required}')
    else:
        version_installed = importlib.metadata.version(base_pkg)
        if REINSTALL_BASE or packaging.version.parse(ver_required) != packaging.version.parse(version_installed):
            run(f'"{python}" -m pip uninstall -y {base_pkg}', f'Uninstall {base_pkg} {version_installed}')
            run(f'"{python}" -m pip install {base_file[platform.system()]}', f'Install {base_pkg} {ver_required}')

    if platform.system() == 'Windows' and is_installed("rembg") and not is_installed("facexlib") and not is_installed("insightface"):
            print(f'Due to Windows restrictions, The new version of SimpleSDXL requires downloading a new installation package, updating the system environment, and then running it. Download URL: https://hf-mirror.com/metercai/SimpleSDXL2/')
            print(f'受组件安装限制，SimpleSDXL2新版本(增加对混元和SD3支持,增加Comfy后端)需要下载新的程序包和基本模型包，在新目录下解压合并目录后再运行。下载地址见：https://hf-mirror.com/metercai/SimpleSDXL2/')
            print(f'If not updated, you can run the commit version using the following scripte: run_SimpleSDXL_commit.bat')
            print(f'如果不升级，可下载SimpleSDXL1的独立分支完全包(未来仅修bug不加功能): https://hf-mirror.com/metercai/SimpleSDXL2/resolve/main/SimpleSDXL1_win64_all.exe.7z; 也可点击run_SimpleSDXL_commit.bat继续运行旧版本(历史存档,无法修bug也不加功能)。')
            sys.exit(0)

    from simpleai_base import simpleai_base
    print("Checking ...")
    token = simpleai_base.init_local(f'SimpleSDXL_User')
    sysinfo = json.loads(token.get_sysinfo().to_json())
    sysinfo.update(dict(did=token.get_did()))
    print(f'[SimpleAI] GPU: {sysinfo["gpu_name"]}, RAM: {sysinfo["ram_total"]}MB, SWAP: {sysinfo["ram_swap"]}MB, VRAM: {sysinfo["gpu_memory"]}MB, DiskFree: {sysinfo["disk_free"]}MB')

    return token, sysinfo

#Intel Arc
#conda install pkg-config libuv
#python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

def prepare_environment():
    global sysinfo

    torch_ver = '2.2.2'
    torchvisio_ver = '0.17.2'
    if sysinfo['gpu_brand'] == 'NVIDIA':
        torch_index_url = "https://download.pytorch.org/whl/cu121"
    elif sysinfo['gpu_brand'] == 'AMD':
        if platform.system() == "Windows":
            #pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
            #pip install torch-directml
            torch_index_url = "https://download.pytorch.org/whl/"
        else:
            torch_index_url = "https://download.pytorch.org/whl/rocm6.0"
            torch_ver = '2.3.1'
            torchvisio_ver = '0.18.1'
    elif sysinfo['gpu_brand'] == 'INTEL':
            torch_index_url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/"
    else:
        torch_index_url = "https://download.pytorch.org/whl/"
    torch_index_url = os.environ.get('TORCH_INDEX_URL', torch_index_url)
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch=={torch_ver} torchvision=={torchvisio_ver} --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
    torch_command += target_path_install
    torch_command += f' -i {index_url} '

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if sysinfo['gpu_brand'] == 'AMD' and platform.system() == "Windows" and not is_installed("torch-directml"):
        run_pip(f"install -U -I --no-deps torch-directml", "torch-directml")

    if TRY_INSTALL_XFORMERS:
        xformers_whl_url_win = 'https://download.pytorch.org/whl/cu121/xformers-0.0.26-cp310-cp310-win_amd64.whl'
        xformers_whl_url_linux = 'https://download.pytorch.org/whl/cu121/xformers-0.0.26-cp310-cp310-manylinux2014_x86_64.whl'
        if not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.26')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_whl_url_win}", "xformers 0.0.26", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_whl_url_linux}", "xformers 0.0.26")
        else:
            version_installed = importlib.metadata.version('xformers')
            if not version_installed.startswith('0.0.26'):
                print(f'Upgrade xformers from {version_installed} to 0.0.26')
                run(f'"{python}" -m pip uninstall -y xformers')
                if platform.system() == "Windows":
                    run_pip(f"install -U -I --no-deps {xformers_whl_url_win}", "xformers 0.0.26")
                elif platform.system() == "Linux":
                    run_pip(f"install -U -I --no-deps {xformers_whl_url_linux}", "xformers 0.0.26")
                else:
                    run_pip(f"install -U -I --no-deps xformers==0.0.26", "xformers 0.0.26")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        if len(met_diff.keys())>0:
            for p in met_diff.keys():
                print(f'Uninstall {p}.{met_diff[p]} ...')
                run(f'"{python}" -m pip uninstall -y {p}=={met_diff[p]}')
        if is_win32_standalone_build:
            run_pip(f"install -r \"{requirements_file}\" -t {target_path_win}", "requirements")
        else:
            run_pip(f"install -r \"{requirements_file}\"", "requirements")

    patch_requirements = "requirements_patch.txt"
    if (REINSTALL_ALL or not requirements_met(patch_requirements)) and not is_win32_standalone_build:
        run_pip(f"install -r \"{patch_requirements}\"", "requirements patching")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


def is_ipynb():
    return True if 'ipykernel' in sys.modules and hasattr(sys, '_jupyter_kernel') else False

build_launcher()
token, sysinfo = check_base_environment()
print(f'[SimpleAI] local_did/本地身份ID: {token.get_did()}')

prepare_environment()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if not args.disable_async_cuda_allocation and args.async_cuda_allocation:
    env_var = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
    if env_var is None:
        env_var = "backend:cudaMallocAsync"
    else:
        env_var += ",backend:cudaMallocAsync"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = env_var


import warnings
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="confy.custom_nodes, hydit, torch.utils")

import socket
if '--listen' not in sys.argv:
    if is_ipynb():
        args.listen = '0.0.0.0'
    else:
        args.listen = socket.gethostbyname(socket.gethostname())
if '--port' not in sys.argv:
    args.port = 6067

if args.hf_mirror is not None : 
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads):
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
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.exists(os.path.join(config.paths_checkpoints[0], default_model)):
            for alternative_model_name in previous_default_models:
                if os.path.exists(os.path.join(config.paths_checkpoints[0], alternative_model_name)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_checkpoints[0], file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_loras[0], file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads)


def reset_env_args():
    global token, sysinfo
    
    sysinfo = json.loads(token.get_sysinfo().to_json())
    sysinfo.update(dict(did=token.get_did()))
    # sysinfo.update(dict(loopback_port=9890))
    #print(f'sysinfo/基础环境信息:{sysinfo}')

    if '--location' in sys.argv:
        sysinfo["location"] = args.location

    if sysinfo["location"] !='CN':
        if '--language' not in sys.argv:
            args.language='default'

    if '--listen' not in sys.argv:
        if is_ipynb():
            args.listen = '127.0.0.1'
        else:
            args.listen = sysinfo["local_ip"]
    if '--port' not in sys.argv:
        args.port = sysinfo["local_port"]

    from enhanced.simpleai import reset_simpleai_args
    reset_simpleai_args(token, sysinfo)

reset_env_args()

from webui import *


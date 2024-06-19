from enum import IntEnum, Enum

disabled = 'Disabled'
enabled = 'Enabled'
subtle_variation = 'Vary (Subtle)'
strong_variation = 'Vary (Strong)'
upscale_15 = 'Upscale (1.5x)'
upscale_2 = 'Upscale (2x)'
upscale_fast = 'Upscale (Fast 2x)'

uov_list = [
    disabled, subtle_variation, strong_variation, upscale_15, upscale_2, upscale_fast
]

CIVITAI_NO_KARRAS = ["euler", "euler_ancestral", "heun", "dpm_fast", "dpm_adaptive", "ddim", "uni_pc"]

# fooocus: a1111 (Civitai)
KSAMPLER = {
    "euler": "Euler",
    "euler_ancestral": "Euler a",
    "heun": "Heun",
    "heunpp2": "",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "",
    "dpmpp_3m_sde_gpu": "",
    "ddpm": "",
    "lcm": "LCM",
    "tcd": "TCD"
}

SAMPLER_EXTRA = {
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": ""
}

SAMPLERS = KSAMPLER | SAMPLER_EXTRA

KSAMPLER_NAMES = list(KSAMPLER.keys())

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "lcm", "turbo", "align_your_steps", "tcd"]
SAMPLER_NAMES = KSAMPLER_NAMES + list(SAMPLER_EXTRA.keys())

sampler_list = SAMPLER_NAMES
scheduler_list = SCHEDULER_NAMES

clip_skip_max = 12

default_vae = 'Default (model)'

refiner_swap_method = 'joint'

cn_ip = "ImagePrompt"
cn_ip_face = "FaceSwap"
cn_canny = "PyraCanny"
cn_cpds = "CPDS"
cn_pose = 'OpenPose'
# cn_depth = "Depth"


# ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face]
ip_list = [cn_ip, cn_canny, cn_cpds, cn_ip_face, cn_pose]
default_ip = cn_ip

# default_parameters = {
#     cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0),
# }  # stop, weight
default_parameters = {
    cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0), cn_pose: (0.926, 1.566),
}  # stop, weight

output_formats = ['png', 'jpeg', 'webp']

inpaint_mask_models = [
    'u2net', 'u2netp', 'u2net_human_seg', 'u2net_cloth_seg', 'silueta', 'isnet-general-use', 'isnet-anime', 'sam'
]

inpaint_mask_cloth_category = ['full', 'upper', 'lower']

inpaint_mask_sam_model = ['sam_vit_b_01ec64', 'sam_vit_h_4b8939', 'sam_vit_l_0b3195']

inpaint_engine_versions = ['None', 'v1', 'v2.5', 'v2.6']
inpaint_option_default = 'Inpaint or Outpaint (default)'
inpaint_option_detail = 'Improve Detail (face, hand, eyes, etc.)'
inpaint_option_modify = 'Modify Content (add objects, change background, etc.)'
inpaint_options = [inpaint_option_default, inpaint_option_detail, inpaint_option_modify]

desc_type_photo = 'Photograph'
desc_type_anime = 'Art/Anime'

translation_timing = ['Translate then generate', 'Modify after translate', 'No translate']
translation_methods = ['Slim Model', 'Big Model', 'Third APIs']

backend_engine_list = ['SDXL', 'SD3 Api', 'SD3Turbo Api']
sdxl_aspect_ratios = [
    '704*1408', '704*1344', '768*1366', '768*1280', '832*1216', '832*1152',
    '896*1152', '896*1088', '915*1144', '960*1024', '960*1088', '1024*1024',
    '1024*960', '1088*960', '1088*896', '1152*896', '1152*832', '1182*886',
    '1216*832', '1254*836', '1280*768', '1344*768', '1344*704', '1366*768',
    '1408*704', '1472*704', '1536*640', '1564*670', '1600*640', '1664*576'
]

class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'
    SIMPLE = 'simple'


metadata_scheme = [
    (f'{MetadataScheme.SIMPLE.value}', MetadataScheme.SIMPLE.value),
    (f'{MetadataScheme.FOOOCUS.value}', MetadataScheme.FOOOCUS.value),
    (f'{MetadataScheme.A1111.value}', MetadataScheme.A1111.value),
]

controlnet_image_count = 4
preparation_step_count = 13


class OutputFormat(Enum):
    PNG = 'png'
    JPEG = 'jpeg'
    WEBP = 'webp'

    @classmethod
    def list(cls) -> list:
        return list(map(lambda c: c.value, cls))


class Steps(IntEnum):
    QUALITY = 60
    SPEED = 30
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class StepsUOV(IntEnum):
    QUALITY = 36
    SPEED = 18
    EXTREME_SPEED = 8
    LIGHTNING = 4
    HYPER_SD = 4


class Performance(Enum):
    QUALITY = 'Quality'
    SPEED = 'Speed'
    EXTREME_SPEED = 'Extreme Speed'
    LIGHTNING = 'Lightning'
    HYPER_SD = 'Hyper-SD'

    @classmethod
    def list(cls) -> list:
        item = list(map(lambda c: c.value, cls))
        item.remove('Extreme Speed')
        return item

    @classmethod
    def has_restricted_features(cls, x) -> bool:
        if isinstance(x, Performance):
            x = x.value
        return x in [cls.EXTREME_SPEED.value, cls.LIGHTNING.value, cls.HYPER_SD.value]
        #return x in [cls.LIGHTNING.value, cls.HYPER_SD.value]

    def steps(self) -> int | None:
        return Steps[self.name].value if Steps[self.name] else None

    def steps_uov(self) -> int | None:
        return StepsUOV[self.name].value if Steps[self.name] else None

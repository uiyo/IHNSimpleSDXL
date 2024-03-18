models = {
    "model": {
        'dir': controlnet_models_path,
        'name': 'thibaud_xl_openpose_256lora.safetensors',
        'loader': 'ControlNet'
    },
    'preprocess_body': {
        'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth',
        'dir': controlnet_models_path,
        'name': 'body_pose_model.pth',
        'loader': 'PoseBody'
    },
    'preprocess_hand': {
        'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth',
        'dir': controlnet_models_path,
        'name': 'hand_pose_model.pth',
        'loader': 'PoseHand'
    },
    'preprocess_face': {
        'url': 'https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth',
        'dir': controlnet_models_path,
        'name': 'facenet.pth',
        'loader': 'PoseFace'
    }
}
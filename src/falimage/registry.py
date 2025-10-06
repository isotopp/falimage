# ------------------------------ Model registry ------------------------------

MODEL_REGISTRY = {
    # Minimal, fast
    "schnell": {
        "endpoint": "fal-ai/flux/schnell",
        "call": "subscribe",
        "allowed": {"prompt", "image_size", "num_images", "num_inference_steps", "enable_safety_checker", "seed"},
        "defaults": {
            "num_inference_steps": 4,
            "enable_safety_checker": False,
        },
    },
    # Dev model supports steps + guidance
    "dev": {
        "endpoint": "fal-ai/flux/dev",
        "call": "subscribe",
        "allowed": {"prompt", "image_size", "num_inference_steps", "guidance_scale", "num_images",
                    "enable_safety_checker", "seed"},
        "defaults": {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "enable_safety_checker": False,
        },
    },
    # Realism model adds strength and output_format
    "realism": {
        "endpoint": "fal-ai/flux-realism",
        "call": "subscribe",
        "allowed": {"prompt", "strength", "image_size", "num_images", "output_format", "num_inference_steps",
                    "guidance_scale", "enable_safety_checker", "seed"},
        "defaults": {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "enable_safety_checker": False,
            "output_format": "jpeg",
            "strength": 1,
        },
    },
    # Bytedance Seedream v4 text-to-image
    "seedream": {
        "endpoint": "fal-ai/bytedance/seedream/v4/text-to-image",
        "call": "subscribe",
        "allowed": {"prompt", "image_size", "num_images", "seed", "enable_safety_checker", "max_images"},
        "defaults": {
            "enable_safety_checker": False,
            "num_images": 1,
        },
    },
    # Bytedance Seedream v4 edit (image editing with multiple reference images)
    "seedream-edit": {
        "endpoint": "fal-ai/bytedance/seedream/v4/edit",
        "call": "subscribe",
        "allowed": {"prompt", "image_size", "num_images", "seed", "enable_safety_checker", "image_urls"},
        "defaults": {
            "enable_safety_checker": False,
            "num_images": 1,
        },
    },
}


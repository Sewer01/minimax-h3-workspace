"""What a fresh install needs: ComfyUI build, node packs, and model variants per GPU class.

Every model has a role (what the workflows use it for). A role can have several
variants; `choose_models` picks one per role from the detected hardware.
"""

COMFY_VERSION = "0.44.0"  # current release; it also brings the newer MiniMax H3 nodes the modes use
MIN_DRIVER = 580  # the bundled PyTorch is built for CUDA 13
COMFY_URL = ("https://github.com/Comfy-Org/ComfyUI/releases/download/"
             f"v{COMFY_VERSION}/ComfyUI_windows_portable_nvidia.7z")
COMFY_SIZE = 2_054_382_457  # the real size of the asset, so the progress bar and the disk estimate do not lie
COMFY_SHA256 = "7170c2db49c182a81adcc051e80234f99dd891656fb200737f87797876348f96"  # GitHub prints this next to the asset
COMFY_DIRNAME = "ComfyUI_windows_portable"

# Nodes the fast RelayVSR upscale needs: they are part of the ComfyUI core, but not of every build,
# so the mode is offered only when the running ComfyUI actually knows them (see App.relay_ready).
RELAY_CLASSES = ("RelayVSRConditioning", "VSRDecode")
LONG_MAX_SCENES = 20   # scenes in one long video; each scene is a separate render (3-15 s), so this is a time budget, not a model limit

# Downloaded as GitHub archives of a fixed commit, so users need no git.
NODE_PACKS = [
    {"name": "ComfyUI-H3-NativeAudioLock", "repo": "SveSop/ComfyUI-H3-NativeAudioLock",
     "commit": "699e58e51b1c8e12b8c0f0dd55da603d870d20e6"},
    {"name": "Comfyui_Minimax_h3_latent_Upscaler", "repo": "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler",
     "commit": "40316cf008b2fd8663263270669eb4da23f89d2c"},
    # separates the voice from music in the lipsync audio editor
    {"name": "ComfyUI-MelBandRoFormer", "repo": "kijai/ComfyUI-MelBandRoFormer",
     "commit": "92c86854e6654f4aacc97484471af95c98ea16d4"},
    # loads reference videos at 24 fps for the references mode
    {"name": "ComfyUI-VideoHelperSuite", "repo": "Kosinkadink/ComfyUI-VideoHelperSuite",
     "commit": "4d907bee61e92c2e65af3bd6383a4e4d356126d1"},
    # memory savers (MiniMaxChunkFeedForward, MiniMaxLowVRAMAttention) used by the long video graphs
    {"name": "ComfyUI-KJNodes", "repo": "kijai/ComfyUI-KJNodes",
     "commit": "ab72eda7ffad6280422d71ab7d54d3fb87c0ba18"},
]

# What each mode needs from custom nodes: checked in a user's own ComfyUI (a new install gets them by itself).
# `classes` are node names ComfyUI must know; `optional` ones only switch a feature on.
NODE_CHECKS = [
    {"pack": "ComfyUI-H3-NativeAudioLock", "classes": ["MiniMaxH3NativeAudioLock"], "modes": ["lipsync", "swap"],
     "optional": False, "why": "точный липсинг под ваше аудио (в замене персонажа звук тоже замораживается этой нодой)"},
    {"pack": "Comfyui_Minimax_h3_latent_Upscaler", "classes": ["MinimaxH3LatentUpscaler3D"], "modes": ["lipsync", "action"],
     "optional": False, "why": "второй проход (увеличение кадра)"},
    {"pack": "ComfyUI-MelBandRoFormer", "classes": ["MelBandRoFormerSampler", "MelBandRoFormerModelLoader"],
     "modes": ["lipsync"], "optional": True, "why": "кнопка «Отделить голос» в редакторе аудио"},
    {"pack": "ComfyUI-VideoHelperSuite", "classes": ["VHS_LoadVideo"], "modes": ["refs", "swap"],
     "optional": False, "why": "видео, в котором меняем персонажа"},
    {"pack": "ComfyUI-KJNodes", "classes": ["MiniMaxChunkFeedForward", "MiniMaxLowVRAMAttention"], "modes": ["long"],
     "optional": False, "why": "экономия видеопамяти в длинном видео"},
]


def node_repo(pack):
    return next(p["repo"] for p in NODE_PACKS if p["name"] == pack)


HF = "https://huggingface.co/{repo}/resolve/main/{path}"
SUB = "MinimaxH3"  # subfolder inside each model category, same as the original workflows
IMG_SUB = "Images"


def _m(repo, path, size_gb, category, name=None, sub=SUB):
    """A downloadable file. `name` is how it is saved (defaults to the file name in the repo)."""
    fname = name or path.rsplit("/", 1)[-1]
    return {"url": HF.format(repo=repo, path=path), "size": int(size_gb * 1e9), "category": category,
            "file": fname, "name": f"{sub}\\{fname}" if sub else fname}


# the same file is published under two names; either one counts as present
FILE_ALIASES = {
    "minimax_h3_latent_upscaler_3d_bf16.safetensors": ("minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",),
}

VARIANTS = {
    "fast_unet": {
        "int8": _m("FastVideo/FastVideo-FastH3-Comfy",
                   "diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", 22.13, "diffusion_models"),
        "nvfp4": _m("FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Comfy",
                    "diffusion_models/fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors", 13.61, "diffusion_models"),
    },
    "lipsync_unet": {
        "full": _m("WarmBloodAban/Minimax-h3_Singularity",
                   "Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors", 34.0, "diffusion_models"),
        "pruned": _m("WarmBloodAban/Minimax-h3_Singularity",
                     "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", 20.97, "diffusion_models"),
        "lite": _m("WarmBloodAban/Minimax-h3_Singularity",
                   "Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors", 11.77, "diffusion_models"),
        # v2: viewers reported that v1.3 generates sound badly
        "v2": _m("WarmBloodAban/Minimax-h3_Singularity",
                 "Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors", 20.97, "diffusion_models"),
        "v2_full": _m("WarmBloodAban/Minimax-h3_Singularity",
                      "Minimax-h3_Singularity_ref2va_v2_int8.safetensors", 34.0, "diffusion_models"),
    },
    "text_encoder": {
        "nvfp4": _m("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    15.69, "text_encoders"),
    },
    "video_vae": {
        "int8": _m("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_int8_convrot.safetensors", 2.81, "vae"),
    },
    "audio_vae": {
        "fp32": _m("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", 0.61, "vae"),
    },
    "ref2v_turbo_lora": {
        "bf16": _m("Comfy-Org/MiniMax-H3", "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                   1.96, "loras"),
    },
    "upscaler": {
        # same file (same sha256) the original workflow knows as minimax_h3_latent_upscaler_3d_bf16
        "bf16": _m("LBH-123-AI/Minimax_h3_latent_Upscaler",
                   "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                   0.69, "latent_upscale_models", name="minimax_h3_latent_upscaler_3d_bf16.safetensors", sub=None),
    },
    # Turbo-SLA (lightx2v): 4-step LoRA for the first/last frame model; used on top of FastH3 for the "fast" switch
    "sla_lora": {
        "bf16": _m("lightx2v/Minimax-h3-Turbo-SLA", "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors",
                   1.96, "loras"),
    },
    "preview": {
        "taeh3": _m("Alissonerdx/Minimax-H3-ComfyUI", "vae_approx/taeh3.safetensors", 0.02, "vae_approx", sub=None),
    },
    # voice / music separation (MelBand RoFormer), used by the audio editor in Lipsync
    "vocal_model": {
        "fp16": _m("Kijai/MelBandRoFormer_comfy", "MelBandRoformer_fp16.safetensors", 0.456, "diffusion_models"),
    },
    # action scenes (Prompt Mastery workflow): FL2VA/Ref2VA hybrid
    "action_unet": {
        "int8": _m("smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models",
                   "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors", 20.97, "diffusion_models"),
        "lite": _m("binglingzhimeng/minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8",
                   "minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8.safetensors", 15.98, "diffusion_models"),
    },
    "speed_slider_lora": {
        "v1.1": _m("RunningHubAI/rh-h3-speed-slider-1.1-lora", "H3_speed_slider_1.1.safetensors", 0.01, "loras"),
    },
    "action_upscaler": {
        "lms": _m("Alissonerdx/Minimax-H3-ComfyUI", "latent_upscaler/h3_upscaler_lms_v0.1.safetensors",
                  0.69, "latent_upscale_models", sub=None),
    },
    # optional LoRAs for the references mode
    "realism_lora": {
        "v1": _m("fal/MiniMax-H3-Realism-People-LoRA", "h3-realism-people-t2v-i2v-r2v.safetensors", 0.13, "loras"),
    },
    "motion_repair_lora": {
        "v1": _m("JOKER141/MiniMax-H3-General-Motion-Continuity-Repair", "Motion_Repair.safetensors", 0.16, "loras"),
    },
    # Character swap (akatz-ai): replaces a person in a video with a character from a picture
    "swap_lora": {
        "v15": _m("akatz-ai/MiniMax-H3-Character-Swap-LoRA", "h3_character_swap_v1.5.safetensors", 0.16, "loras"),
    },
    # Pika H3 Flex Turbo: 8-step HD LoRA for references; a viewer favourite for the generated sound
    "pika_lora": {
        "hd8": _m("Pika-Labs/Pika-H3-Flex-Turbo-LoRA", "pika_h3_flex_turbo_8step_hd_ref2va.safetensors", 3.91, "loras"),
    },
    # improve quality of a finished video (SeedVR2, plain ComfyUI nodes)
    "seedvr_unet": {
        "int8": _m("Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", 3.46,
                   "diffusion_models", sub=IMG_SUB),
    },
    "seedvr_vae": {
        "fp16": _m("Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", 0.5, "vae", sub=IMG_SUB),
    },
    # fast video enlarge x2 in one sampling step (RelayVSR, built into ComfyUI); video only
    "relayvsr_unet": {
        "x2": _m("Comfy-Org/RelayVSR", "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors", 5.16,
                 "diffusion_models", sub=IMG_SUB),
    },
    "relayvsr_vae": {
        "x2": _m("Comfy-Org/RelayVSR", "vae/relayvsr_flash_vae_x2_fp32.safetensors", 0.13, "vae", sub=IMG_SUB),
    },
    # pictures: Krea 2 (by text) and Qwen Image 2.1 (edit a picture, character sheet); both are plain ComfyUI nodes
    "krea_unet": {
        "int8": _m("Comfy-Org/Krea-2", "diffusion_models/krea2_turbo_int8_convrot.safetensors", 13.49,
                   "diffusion_models", sub=IMG_SUB),
    },
    "krea_clip": {
        "fp8": _m("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5.24,
                  "text_encoders", sub=IMG_SUB),
    },
    "krea_vae": {
        "bf16": _m("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", 0.25, "vae", sub=IMG_SUB),
    },
    # Krea 2 for realistic people and photos: RedCraft RedMix + the Realism Engine LoRA (two passes)
    "kreareal_unet": {
        "fp8": _m("Kerstal/krea-2_model", "redcraftKREA2RedMix_krea2Edition.safetensors", 13.14,
                  "diffusion_models", sub=IMG_SUB),
    },
    "kreareal_lora": {
        "v2": _m("uzumix/realism_engine_krea2_v2", "realism_engine_krea2_v2.safetensors", 1.56, "loras", sub=IMG_SUB),
    },
    "qwen_unet": {
        "int8": _m("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", 7.26,
                   "diffusion_models", sub=IMG_SUB),
    },
    "qwen_clip": {
        "int8": _m("Comfy-Org/Qwen-Image-2.1", "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9.35,
                   "text_encoders", sub=IMG_SUB),
    },
    "qwen_vae": {
        "bf16": _m("Comfy-Org/Qwen-Image-2.1", "vae/qwen_image_2.1_vae_bf16.safetensors", 0.68, "vae", sub=IMG_SUB),
    },
    # FLUX.2 klein 4B (Apache 2.0, free for commercial use): edit a picture by text
    "flux_unet": {
        "fp8": _m("black-forest-labs/FLUX.2-klein-4b-fp8", "flux-2-klein-4b-fp8.safetensors", 4.07,
                  "diffusion_models", sub=IMG_SUB),
    },
    "flux_clip": {
        "bf16": _m("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", 8.04,
                   "text_encoders", sub=IMG_SUB),
    },
    "flux_vae": {
        "bf16": _m("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", 0.34, "vae", sub=IMG_SUB),
    },
}

# sha256 of every model file this program downloads, exactly as Hugging Face publishes it
# (the LFS oid of the file, read through their API - nothing was downloaded to build this).
# A mirror or a damaged CDN answer must never quietly become someone else's weights.
# Files that Hugging Face keeps as plain git objects have no published sum and are not here.
MODEL_SHA256 = {
    "Alissonerdx/Minimax-H3-ComfyUI|latent_upscaler/h3_upscaler_lms_v0.1.safetensors": "40d4228227146245ef2b65480b647c50222465ca125de5880f192e3259e40d59",
    "Alissonerdx/Minimax-H3-ComfyUI|vae_approx/taeh3.safetensors": "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13",
    "Comfy-Org/Krea-2|diffusion_models/krea2_turbo_int8_convrot.safetensors": "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1",
    "Comfy-Org/Krea-2|text_encoders/qwen3vl_4b_fp8_scaled.safetensors": "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
    "Comfy-Org/Krea-2|vae/qwen_image_vae.safetensors": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    "Comfy-Org/MiniMax-H3|loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
    "Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    "Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    "Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_int8_convrot.safetensors": "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e",
    "Comfy-Org/Qwen-Image-2.1|diffusion_models/qwen_image_2.1_int8_convrot.safetensors": "cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d",
    "Comfy-Org/Qwen-Image-2.1|text_encoders/qwen3vl_8b_int8_convrot.safetensors": "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
    "Comfy-Org/Qwen-Image-2.1|vae/qwen_image_2.1_vae_bf16.safetensors": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
    "Comfy-Org/RelayVSR|diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors": "240e75e0c17cf2845d559b25254a230ff7458f81449d7a8c5e7e3fb8275c7b3c",
    "Comfy-Org/RelayVSR|vae/relayvsr_flash_vae_x2_fp32.safetensors": "b5bb0339acb5b52545911ea29a51b4c5b805776a779222b5a14591571f42e21c",
    "Comfy-Org/SeedVR2|diffusion_models/seedvr2_3b_int8_convrot.safetensors": "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157",
    "Comfy-Org/SeedVR2|vae/seedvr2_ema_vae_fp16.safetensors": "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    "Comfy-Org/flux2-dev|split_files/vae/flux2-vae.safetensors": "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
    "Comfy-Org/z_image_turbo|split_files/text_encoders/qwen_3_4b.safetensors": "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
    "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Comfy|diffusion_models/fastvideo_fasth3_8step_v2_pruned_nvfp4.safetensors": "49371f21b546e59813984938a1b4fbea9348eb5003e07a01d48083efcc5a1d94",
    "FastVideo/FastVideo-FastH3-Comfy|diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors": "0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8",
    "JOKER141/MiniMax-H3-General-Motion-Continuity-Repair|Motion_Repair.safetensors": "a678a3657cefd5c2c2712f892da2f71be53c3f252e335557e918854f8abc579f",
    "Kerstal/krea-2_model|redcraftKREA2RedMix_krea2Edition.safetensors": "92e8d4eecbf4f89b30140dc231e377118823be3a30c5c04342bcf394fe27e73b",
    "Kijai/MelBandRoFormer_comfy|MelBandRoformer_fp16.safetensors": "6119aef379a6c7264e0b37db65ae1e6488b8ca4a00baf56d6d244737b8488226",
    "LBH-123-AI/Minimax_h3_latent_Upscaler|minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors": "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
    "Pika-Labs/Pika-H3-Flex-Turbo-LoRA|pika_h3_flex_turbo_8step_hd_ref2va.safetensors": "dfe46d972408d1094053ff1f30c720bf16176ec334962009767ab0fff29e6de2",
    "RunningHubAI/rh-h3-speed-slider-1.1-lora|H3_speed_slider_1.1.safetensors": "593b074e8eb17bf9f858c3586fc12f53038dbfdd9634e146fe34a17e2661ca40",
    "WarmBloodAban/Minimax-h3_Singularity|Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors": "412a7b126595a958964193f3b42513d7cf2df4196ef04223a0f05e3622949cce",
    "WarmBloodAban/Minimax-h3_Singularity|Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors": "2a319b6fafbd646c123ac12a28b1908c7468b148a8c03164172b2f72370264ec",
    "WarmBloodAban/Minimax-h3_Singularity|Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors": "551915097b8727537a82f3a2c5acec965165ec84e780b3e9a5f28ce867fddf08",
    "WarmBloodAban/Minimax-h3_Singularity|Minimax-h3_Singularity_ref2va_v2_int8.safetensors": "db3ec568863db6f5ec1840210d41653afef73d4df7b0cb489c52ad458b6f8a7f",
    "WarmBloodAban/Minimax-h3_Singularity|Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors": "d7637c2c35cd30facd9e93778c491f6d82f91c4a441aecbdb4c384a62a5f3d9f",
    "akatz-ai/MiniMax-H3-Character-Swap-LoRA|h3_character_swap_v1.5.safetensors": "8f2c3c3eba161df8d0c6099598b4730f641f5ce01e04cd16d2427cff8a35218f",
    "binglingzhimeng/minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8|minimax_h3_hybrid_fl2va_ref2va_b25-49_w6a8.safetensors": "156f0c302593777c25b49d87565b02239715b128595128ef683a759ee6ab0406",
    "black-forest-labs/FLUX.2-klein-4b-fp8|flux-2-klein-4b-fp8.safetensors": "97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6",
    "fal/MiniMax-H3-Realism-People-LoRA|h3-realism-people-t2v-i2v-r2v.safetensors": "acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e",
    "lightx2v/Minimax-h3-Turbo-SLA|minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors": "5cae6df40a06ea825f85fc8876c9ea1c9692c833a9af07bb8b3bac9ce2a71bac",
    "smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models|minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors": "a629cfea8d89a071b140c6e1935dc9a23e72de6badc18975a2bb9e6d1423d76d",
    "uzumix/realism_engine_krea2_v2|realism_engine_krea2_v2.safetensors": "1815c6ac01216ad62f8dd17edab7c1fd31edf16c1b11434317123bae60e33257",
}


LICENSES = {
    # Классы: nc — нельзя в коммерции; h3 — MiniMax H3 Community License (коммерция разрешена, но
    # продукт обязан показывать «MiniMax H3» на интерфейсе, при доходе свыше 20 млн $ в год нужно
    # письменное разрешение MiniMax, на результатах нельзя обучать чужие модели); mixed — ролики
    # ваши, а саму модель как сервис не сдавать; unknown — в карточке лицензии нет.
    "Comfy-Org/Qwen-Image-2.1": ("nc", "Qwen Image 2.1 — исследовательская лицензия: только личное использование"),
    "Comfy-Org/flux2-dev": ("nc", "FLUX.2 dev — некоммерческая лицензия"),
    "Comfy-Org/Krea-2": ("unknown", "Krea 2 Community License — условия в PDF, перед коммерцией проверьте"),
    "uzumix/realism_engine_krea2_v2": ("unknown", "лицензия не указана — для коммерции уточните у автора"),
    "RunningHubAI/rh-h3-speed-slider-1.1-lora": ("unknown", "лицензия не указана — для коммерции уточните у автора"),
    "Kijai/MelBandRoFormer_comfy": ("unknown", "лицензия не указана — для коммерции уточните у автора"),
    "Alissonerdx/Minimax-H3-ComfyUI": ("mixed", "ролики можно и в коммерции; саму LoRA как сервис не сдавать"),
    "noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF": ("unknown", "производная модель без указанной лицензии"),
}
H3_NOTE = "MiniMax H3: в коммерческом продукте на интерфейсе должно быть указано «MiniMax H3»"
LIC_BADGE = {"nc": "нельзя в коммерцию", "unknown": "лицензия не указана",
             "mixed": "ролик можно, модель — нет", "h3": "MiniMax H3: условия"}


def license_of(url):
    """(класс, пояснение) для файла по его адресу. Пустой класс — свободная лицензия (Apache/MIT)."""
    if "huggingface.co/" not in (url or ""):
        return "", ""
    repo = url.split("huggingface.co/", 1)[1].split("/resolve/", 1)[0]
    if repo in LICENSES:
        return LICENSES[repo]
    low = repo.lower()
    if "minimax" in low or "-h3" in low or "h3-" in low or low.endswith("h3"):
        return "h3", H3_NOTE
    return "", ""


def file_sha256(url):
    """The sha256 Hugging Face publishes for this model file, or "" when we have none."""
    if "huggingface.co/" not in url or "/resolve/" not in url:
        return ""
    try:
        repo, tail = url.split("huggingface.co/", 1)[1].split("/resolve/", 1)
        _rev, path = tail.split("/", 1)
    except ValueError:
        return ""
    return MODEL_SHA256.get(f"{repo}|{path}", "")


# What a file name of each role must contain (any of these words, any case). Used only to warn when a file is picked
# by hand that does not look like the right one, and to say what to check when ComfyUI fails with a shape error.
ROLE_MARKS = {
    "fast_unet": ["fasth3", "h3"], "lipsync_unet": ["singularity", "h3"], "text_encoder": ["32b"],
    "video_vae": ["video_vae"], "audio_vae": ["audio_vae"], "ref2v_turbo_lora": ["turbo"],
    "upscaler": ["upscaler"], "preview": ["taeh3"], "action_unet": ["hybrid", "fl2va"],
    "speed_slider_lora": ["speed"], "action_upscaler": ["upscaler"], "vocal_model": ["melband"],
    "realism_lora": ["realism"], "motion_repair_lora": ["motion"], "pika_lora": ["pika"], "swap_lora": ["swap"], "sla_lora": ["sla"],
    "seedvr_unet": ["seedvr"], "seedvr_vae": ["seedvr"], "relayvsr_unet": ["relayvsr"], "relayvsr_vae": ["relayvsr"], "krea_clip": ["qwen3vl", "qwen3-vl"],
    "krea_vae": ["qwen_image", "qwen-image"], "kreareal_unet": ["krea"], "kreareal_lora": ["realism"],
    "qwen_unet": ["qwen"], "qwen_clip": ["qwen3vl", "qwen3-vl"], "qwen_vae": ["qwen"],
    "flux_unet": ["flux"], "flux_clip": ["qwen_3", "qwen3"], "flux_vae": ["flux"],
}


def looks_right(role, filename):
    """False only when the role has a name rule and the file name breaks it."""
    marks = ROLE_MARKS.get(role)
    n = filename.replace("/", "\\").split("\\")[-1].lower()
    return not marks or any(m in n for m in marks)


def marks_text(role):
    return " или ".join(f"«{m}»" for m in ROLE_MARKS.get(role, [])[:2])


# Plain words for the "another version" choice in the Models section (only versions of the same model replace each other)
VARIANT_LABELS = {
    "fast_unet": {"int8": "обычная (int8)", "nvfp4": "лёгкая NVFP4 — только для видеокарт RTX 50-й серии"},
    "lipsync_unet": {"full": "v1.3 полная (самая тяжёлая)", "pruned": "v1.3 облегчённая", "lite": "v1.3 лёгкая (слабее качество)",
                     "v2": "v2 облегчённая — по отзывам, звук лучше", "v2_full": "v2 полная"},
    "action_unet": {"int8": "обычная (int8)", "lite": "лёгкая (меньше памяти, чуть слабее)"},
}

# LoRA from the original action workflow that has no public download; used when the user has it
ACTION_DMAD_LORA = "minimax_h3_dmad_full_dareties_v4_step600_comfy_fro"

ROLE_TITLES = {
    "fast_unet": "Основная модель — Image-to-Video (FastH3)",
    "lipsync_unet": "Основная модель — Lipsync (Singularity)",
    "text_encoder": "Текстовый энкодер (Qwen3-VL 32B)",
    "video_vae": "Видео VAE",
    "audio_vae": "Аудио VAE",
    "ref2v_turbo_lora": "LoRA ускорения для Lipsync",
    "upscaler": "Апскейлер второго прохода",
    "preview": "Декодер живого превью",
    "action_unet": "Основная модель — Экшн-сцены (Hybrid FL2VA/Ref2VA)",
    "speed_slider_lora": "LoRA Speed Slider (экшн)",
    "action_upscaler": "Апскейлер SelfLift (экшн, точный режим)",
    "vocal_model": "Отделение голоса от музыки (MelBand RoFormer)",
    "realism_lora": "LoRA «Реализм людей» (по желанию)",
    "motion_repair_lora": "LoRA «Исправление движения» (по желанию)",
    "swap_lora": "LoRA «Замена персонажа» (akatz-ai)",
    "pika_lora": "LoRA Pika H3 Flex Turbo (ускорение для Референсов, хороший звук)",
    "sla_lora": "LoRA Turbo-SLA (быстрое ускорение для Image-to-Video)",
    "seedvr_unet": "Увеличение видео и картинок — модель SeedVR2 (3B)",
    "seedvr_vae": "Увеличение видео и картинок — VAE SeedVR2",
    "relayvsr_unet": "Быстрое увеличение видео — модель RelayVSR ×2",
    "relayvsr_vae": "Быстрое увеличение видео — VAE RelayVSR ×2",
    "krea_unet": "Основная модель — Картинка по тексту (Krea 2)",
    "krea_clip": "Текстовый энкодер Krea 2 (Qwen3-VL 4B)",
    "krea_vae": "VAE для Krea 2",
    "kreareal_unet": "Основная модель — Картинка: реализм (RedCraft, Krea 2)",
    "kreareal_lora": "LoRA Realism Engine для Krea 2 (реализм)",
    "qwen_unet": "Основная модель — Правка картинки (Qwen Image 2.1)",
    "qwen_clip": "Текстовый энкодер Qwen (Qwen3-VL 8B)",
    "qwen_vae": "VAE для Qwen Image 2.1",
    "flux_unet": "Основная модель — Правка картинки для коммерции (FLUX.2 klein 4B)",
    "flux_clip": "Текстовый энкодер FLUX.2 klein (Qwen3 4B)",
    "flux_vae": "VAE для FLUX.2",
}

MODE_TITLES = {"video": "Image-to-Video", "lipsync": "Lipsync", "refs": "Референсы", "actor": "Актёрская игра", "swap": "Замена персонажа", "action": "Экшн-сцены",
               "krea": "Картинка по тексту", "kreareal": "Картинка: реализм", "qwen": "Правка картинки", "flux": "Правка картинки: FLUX.2 (можно в коммерцию)",
               "upscale": "Увеличение видео и картинок", "relayvsr": "Быстрое увеличение видео (RelayVSR)", "long": "Длинное видео"}

# Which roles each mode needs.
MODE_ROLES = {
    "video": ["fast_unet", "text_encoder", "video_vae", "audio_vae"],
    "lipsync": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora", "upscaler"],
    "refs": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora"],
    # same models as references: dialogue scenes with a voice sample, acting-first prompts
    "actor": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora"],
    "action": ["action_unet", "text_encoder", "video_vae", "audio_vae", "upscaler"],
    # character swap preset: same Ref2V model as References + a small swap LoRA
    "swap": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "ref2v_turbo_lora", "swap_lora"],
    # long video / clips: scenes that continue each other (Singularity Ref2V + Pika LoRA, same files as References)
    "long": ["lipsync_unet", "text_encoder", "video_vae", "audio_vae", "pika_lora"],
    "krea": ["krea_unet", "krea_clip", "krea_vae"],
    "kreareal": ["kreareal_unet", "kreareal_lora", "krea_clip", "krea_vae"],
    "qwen": ["qwen_unet", "qwen_clip", "qwen_vae"],
    "flux": ["flux_unet", "flux_clip", "flux_vae"],
    "upscale": ["seedvr_unet", "seedvr_vae"],
    "relayvsr": ["relayvsr_unet", "relayvsr_vae"],
}
# Small extras downloaded with a mode but not required to run it.
MODE_OPTIONAL = {
    "video": ["sla_lora"],
    "lipsync": ["realism_lora", "vocal_model"],
    "refs": ["pika_lora", "realism_lora", "motion_repair_lora"],
    "actor": ["realism_lora"],
    "swap": ["pika_lora", "realism_lora", "motion_repair_lora"],
    "action": ["speed_slider_lora", "action_upscaler"],
    "long": ["vocal_model"],
}
ALL_ROLES = list(VARIANTS)

# The user's original files, for connecting to an existing ComfyUI that already has them.
DEFAULT_CHOICE = {"fast_unet": "int8", "lipsync_unet": "full", "text_encoder": "nvfp4", "video_vae": "int8",
                  "audio_vae": "fp32", "ref2v_turbo_lora": "bf16", "upscaler": "bf16", "preview": "taeh3",
                  "action_unet": "int8", "speed_slider_lora": "v1.1", "action_upscaler": "lms", "vocal_model": "fp16",
                  "realism_lora": "v1", "motion_repair_lora": "v1", "pika_lora": "hd8", "swap_lora": "v15", "sla_lora": "bf16",
                  "krea_unet": "int8", "krea_clip": "fp8", "krea_vae": "bf16", "kreareal_unet": "fp8", "kreareal_lora": "v2",
                  "qwen_unet": "int8", "qwen_clip": "int8", "qwen_vae": "bf16",
                  "flux_unet": "fp8", "flux_clip": "bf16", "flux_vae": "bf16",
                  "seedvr_unet": "int8", "seedvr_vae": "fp16", "relayvsr_unet": "x2", "relayvsr_vae": "x2"}


PROFILES = {
    "lite": {"title": "Лёгкий", "lipsync_unet": "lite", "action_unet": "lite",
             "about": "Самые лёгкие версии: меньше качать и меньше памяти, качество немного ниже"},
    "standard": {"title": "Стандарт", "lipsync_unet": "v2", "action_unet": "int8",
                 "about": "Облегчённая (pruned) Singularity v2: ~20 ГБ вместо 34, качество почти как у полной. Подходит для видеокарт 12–16 ГБ"},
    "max": {"title": "Максимум", "lipsync_unet": "full", "action_unet": "int8",
            "about": "Полные модели, как в исходных workflow"},
}


def recommended_profile(hw):
    vram, ram = hw.get("vram_gb") or 0, hw.get("ram_gb") or 0
    if vram + ram >= 48 and vram >= 24:  # "max" puts the full ~32 GB Singularity: too heavy for 16 GB cards
        return "max"
    if vram + ram >= 40:
        return "standard"
    return "lite"


def roles_for(modes, optional=True):
    # the live-preview decoder (taeh3) is no longer added here: server.preview_opts() keeps previews off, and a
    # file nothing uses must not make the setup say "models are missing" nor force a download
    roles = {r for m in modes for r in MODE_ROLES[m]}
    if optional:
        roles |= {r for m in modes for r in MODE_OPTIONAL.get(m, [])}
    return [r for r in ALL_ROLES if r in roles]


def plan_size(choice, modes):
    return sum(VARIANTS[r][choice[r]]["size"] for r in roles_for(modes))


def choose_models(hw, profile=None):
    """Pick a variant per role for this computer. Returns (choice, notes, verdict, profile).

    ComfyUI keeps what does not fit in VRAM in system RAM, so the real budget is VRAM + RAM.
    """
    vram, ram = hw.get("vram_gb") or 0, hw.get("ram_gb") or 0
    blackwell = (hw.get("compute_cap") or 0) >= 12
    profile = profile if profile in PROFILES else recommended_profile(hw)
    choice = dict(DEFAULT_CHOICE, lipsync_unet=PROFILES[profile]["lipsync_unet"],
                  action_unet=PROFILES[profile]["action_unet"])
    notes = []

    if blackwell and vram >= 16:
        choice["fast_unet"] = "nvfp4"
        notes.append("Видеокарта RTX 50-й серии: для генерации видео берём версию NVFP4 — она легче и быстрее на этих картах.")

    if hw.get("nvidia") and 0 < (hw.get("driver") or 0) < MIN_DRIVER:
        notes.insert(0, f"Драйвер видеокарты устарел ({hw['driver']:g}). Нужен {MIN_DRIVER} или новее, иначе ComfyUI не запустится — обновите драйвер.")

    cap = hw.get("compute_cap") or 0
    if not hw.get("nvidia"):
        verdict = "unsupported"
        notes.insert(0, "Приложение работает только с видеокартами NVIDIA (RTX 20-й серии и новее, от 8 ГБ памяти). "
                        "Видеокарты AMD Radeon, встроенная графика Intel и компьютеры Mac не поддерживаются.")
    elif 0 < cap < 7.5:
        verdict = "unsupported"
        notes.insert(0, "Ваша видеокарта слишком старая для этой программы. Нужна NVIDIA RTX 20-й серии или новее "
                        "(например, RTX 2060, 3060, 4060) с памятью от 8 ГБ.")
    elif vram < 8:
        verdict = "unsupported"
        notes.insert(0, f"Видеопамяти {vram:g} ГБ — для работы нужно минимум 8 ГБ (лучше 16). "
                        "На такой видеокарте генерация видео не поместится в память.")
    elif vram < 12 or ram < 32:
        verdict = "weak"
        if vram < 12:
            notes.insert(0, f"Видеопамяти {vram:g} ГБ — генерация будет медленной, лучше выбирать 480p или 720p.")
        if ram < 32:
            notes.insert(0, f"Оперативной памяти {ram:g} ГБ, а рекомендуется 32 ГБ: модели могут не поместиться, "
                            "Windows начнёт использовать файл подкачки и всё сильно замедлится.")
    else:
        verdict = "ok"
    return choice, notes, verdict, profile


def model_entry(role, variant):
    return dict(VARIANTS[role][variant], role=role, variant=variant, title=ROLE_TITLES[role])


# modes that appeared after the first release: the app offers them once after an update
OFFER_MODES = ["flux", "long"]

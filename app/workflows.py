"""ComfyUI API graphs for Minimax H3 Workspace modes.

Each builder returns (graph, stages): the API-format prompt and a map
node_id -> UI stage name, so progress can be shown in plain words.
The graphs mirror the user's workflows in the project root.
"""

import math

FPS = 24

# Model files come in as `m`: role -> name as ComfyUI lists it (see catalog.py).

ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4)}
QUALITY = {"480p": 864 * 480, "720p": 1280 * 736, "HD+": 1376 * 768, "1080p": 1920 * 1088}


def r32(v):
    return max(32, round(v / 32) * 32)


def resolution(aspect, quality):
    w, h = ASPECTS[aspect]
    px = QUALITY[quality]
    return r32(math.sqrt(px * w / h)), r32(math.sqrt(px * h / w))


def frames(seconds):
    # same formula as the workflow's ComfyMathExpression: 24 fps, length = 17k + 5
    f = max(5, round(seconds * FPS))
    return f + (5 - f % 17) % 17


def video_graph(m, prompt, width, height, seconds, seed, first=None, last=None):
    """MINIMAXH3_FASTvideo_I2V: text / first frame / first + last frame."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["fast_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["1", 0], "shift_video": 10, "shift_audio": 3}},
        "3": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["2", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["3", 0], "selection": "vsa", "selection.keep_percent": 10.0,
            "start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "",
            "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}},
        "5": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds)}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["4", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["12", 0], "sampler": ["10", 0],
            "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0], "audio": ["15", 0], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "17": {"class_type": "SaveVideo", "inputs": {
            "video": ["16", 0], "filename_prefix": "H3Workspace/video",
            "format": "auto", "format.codec": "auto"}},
    }
    for slot, (load_id, scale_id, image) in {
        "first_frame": ("20", "21", first), "last_frame": ("22", "23", last)}.items():
        if not image:
            continue
        g[load_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g[scale_id] = {"class_type": "ImageScale", "inputs": {
            "image": [load_id, 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center"}}
        g["8"]["inputs"][slot] = [scale_id, 0]

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "encode" for n in ("8", "20", "21", "22", "23")})
    stages.update({"13": "sample", "14": "decode", "15": "decode", "16": "save", "17": "save"})
    return g, stages


def _refs(g, cond_node, images, videos=(), audios=(), base=100):
    """LoadImage / VHS_LoadVideo (24 fps) / LoadAudio nodes wired into MiniMaxH3ReferenceToVideo."""
    inputs = g[cond_node]["inputs"]
    for i, name in enumerate(images[:9]):
        nid = str(base + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        inputs[f"ref_images.ref_image_{i}"] = [nid, 0]
    for i, name in enumerate(videos[:3]):
        nid = str(base + 10 + i)
        g[nid] = {"class_type": "VHS_LoadVideo", "inputs": {
            "video": name, "force_rate": 24, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": 360, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
        inputs[f"ref_videos.ref_video_{i}"] = [nid, 0]
    for i, name in enumerate(audios[:3]):
        nid = str(base + 20 + i)
        g[nid] = {"class_type": "LoadAudio", "inputs": {"audio": name}}
        inputs[f"ref_audios.ref_audio_{i}"] = [nid, 0]
    return [str(base + i) for i in range(len(images[:9]))] + [str(base + 10 + i) for i in range(len(videos[:3]))] \
        + [str(base + 20 + i) for i in range(len(audios[:3]))]


def _lora_chain(g, start, loras, base=60):
    """Stack LoraLoaderModelOnly nodes; returns the last model output."""
    prev = start
    for i, (name, strength) in enumerate(loras):
        nid = str(base + i)
        g[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": prev, "lora_name": name, "strength_model": strength}}
        prev = [nid, 0]
    return prev


def ref_graph(m, prompt, width, height, seconds, seed, images, videos, audios, steps=6,
              ref_size="max", extra_loras=()):
    """MinimaxH3_Ref to Video (ULTRA SPEED Singularity 4-step), single pass.

    The original samples with T8's Dual Clock node (shift video 12 / audio 4, beta);
    here the same shifts go through ComfyUI's built-in MiniMaxH3SigmaShift + beta scheduler.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
    }
    model = _lora_chain(g, ["2", 0], list(extra_loras) + [(m["ref2v_turbo_lora"], 0.7)])
    g.update({
        "9": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["9", 0], "scheduler": "beta", "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/refs", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, videos, audios)
    stages = {n: "load" for n in ["1", "2", "4", "5", "6"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update({"14": "sample", "15": "decode", "16": "decode", "17": "save", "18": "save"})
    return g, stages


ACTION_LOWRES = 0.6  # SelfLift lowres_scale in the original


def action_graph(m, prompt, width, height, seconds, seed, images, audios, faithful, loras):
    """PM Minimax H3 Best Action Workflow (Prompt Mastery): hybrid model, sol-attn, 8 low-res + 4 high-res steps.

    faithful=True runs the original SelfLiftH3Sampler + H3SigmaRefiner; otherwise the same schedule is
    done with built-in nodes: pass 1 at 0.6x, learned latent upscale, pass 2 at full size.
    width/height are the final size.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["action_unet"], "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
    }
    model = _lora_chain(g, ["1", 0], loras)  # dmad 0.8, speed slider 4 in the original
    g.update({
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    })
    if faithful:
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": width, "height": height, "length": frames(seconds), "ref_image_size": "max"}},
            "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
            "13": {"class_type": "H3SigmaRefiner", "inputs": {
                "sigmas": ["10", 0], "extra_steps": 1, "strength": 0.7, "offset": 0.0, "curve": "cosine"}},
            "14": {"class_type": "SelfLiftH3Sampler", "inputs": {
                "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0], "vae": ["5", 0],
                "latent_image": ["8", 1], "sampler": ["12", 0], "sigmas": ["13", 0], "seed": seed, "cfg": 1.0,
                "transition_step": 8, "lowres_scale": ACTION_LOWRES, "rho": 0.0, "w_min": 0.5, "w_max": 1.0,
                "upscaler_model": m["action_upscaler"], "highres_tiling": True}},
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample"}
    else:
        lw, lh = r32(width * ACTION_LOWRES), r32(height * ACTION_LOWRES)
        g.update({
            "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
                "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
                "width": lw, "height": lh, "length": frames(seconds), "ref_image_size": "max"}},
            "13": {"class_type": "BasicGuider", "inputs": {"model": ["7", 0], "conditioning": ["8", 0]}},
            "9": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
            "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}},
            "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 1]}},
            "16": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
                "latent": ["15", 0], "model_name": m["upscaler"], "mode": "scale by multiplier",
                "mode.scale": round(1 / ACTION_LOWRES, 3), "align": 32, "enable_temporal_chunking": True,
                "force_unload": True, "device": "cuda", "precision": "fp16"}},
            "17": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["16", 0], "audio_latent": ["15", 1]}},
            "18": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["9", 1], "latent_image": ["17", 0]}},
            "20": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 1], "vae": ["5", 0]}},
            "21": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["18", 1], "vae": ["6", 0]}},
        })
        sample_stages = {"14": "sample", "15": "upscale", "16": "upscale", "17": "upscale", "18": "sample2"}
    g.update({
        "22": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["21", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "23": {"class_type": "SaveVideo", "inputs": {
            "video": ["22", 0], "filename_prefix": "H3Workspace/action", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, (), audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
    stages.update({n: "encode" for n in ["8"] + loaders})
    stages.update(sample_stages)
    stages.update({"20": "decode", "21": "decode", "22": "save", "23": "save"})
    return g, stages


def lipsync_scale(quality):
    return 2.0 if quality == "1080p" else 1.5


def lipsync_prompt(text, n_refs):
    """Wrap free text into H3's structured prompt unless the user already wrote one."""
    if "integrated_multimodal_description" in text:
        return text
    refs = " ".join(f"<Picture {i}> (from [Shot 1]) is fully referenced." for i in range(1, n_refs + 1))
    return (f"For the target video, at 0.00 seconds into the target video, {refs}\n\n"
            f"integrated_multimodal_description: [Shot 1] {text}\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A")


def lipsync_graph(m, prompt, width, height, seconds, seed, audio, ref_images, scale, ref_size="max", extra_loras=()):
    """MiniMax H3 Singularity Ref2V + Audio Lock, two passes.

    width/height are pass 1 size; pass 2 upscales the latent by `scale`.
    `audio` is the already trimmed clip; the video length follows it.
    ref_size: "max" (best identity, several times slower) or "match" (refs scaled to the video area).
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        # the original workflow uses KJNodes' SageAttention patch; the built-in INT8 attention
        # needs no extra install (sageattention is hard to set up on Windows)
        "2": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": m["ref2v_turbo_lora"], "strength_model": 0.55}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": ref_size}},
        "9": {"class_type": "MiniMaxH3NativeAudioLock", "inputs": {
            "model": ["1", 0], "av_latent": ["8", 1], "audio_vae": ["6", 0], "audio": ["7", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {
            "model": ["9", 0], "scheduler": "beta", "steps": 12, "denoise": 1.0}},
        "11": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["10", 0], "step": 8}},
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["8", 0]}},
        # pass 1
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 0], "latent_image": ["9", 1]}},
        # pass 2: upscale the video latent, keep the audio latent, finish the low sigmas
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 1]}},
        "17": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": ["16", 0], "model_name": m["upscaler"],
            "mode": "scale by multiplier", "mode.scale": scale, "align": 32,
            "enable_temporal_chunking": True, "force_unload": True,
            "device": "cuda", "precision": "fp16"}},
        "18": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["17", 0], "audio_latent": ["16", 1]}},
        "19": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["12", 0], "guider": ["14", 0], "sampler": ["13", 0],
            "sigmas": ["11", 1], "latent_image": ["18", 0]}},
        "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 1], "vae": ["5", 0]}},
        # the saved video carries the original audio, not a regenerated one
        "21": {"class_type": "CreateVideo", "inputs": {
            "images": ["20", 0], "audio": ["9", 2], "fps": float(FPS),
            "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "22": {"class_type": "SaveVideo", "inputs": {
            "video": ["21", 0], "filename_prefix": "H3Workspace/lipsync",
            "format": "auto", "format.codec": "auto"}},
    }
    for i, image in enumerate(ref_images[:9]):
        node_id = str(30 + i)
        g[node_id] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g["8"]["inputs"][f"ref_images.ref_image_{i}"] = [node_id, 0]
    if extra_loras:  # optional LoRAs go in front of the speed LoRA, as in the original workflow
        g["3"]["inputs"]["model"] = _lora_chain(g, ["2", 0], list(extra_loras))

    stages = {n: "load" for n in ("1", "2", "3", "4", "5", "6", "7")}
    stages.update({n: "load" for n in g if n.isdigit() and 60 <= int(n) < 70})
    stages.update({n: "encode" for n in ["8", "9"] + [str(30 + i) for i in range(9)]})
    stages.update({"15": "sample", "16": "upscale", "17": "upscale", "18": "upscale",
                   "19": "sample2", "20": "decode", "21": "save", "22": "save"})
    return g, stages

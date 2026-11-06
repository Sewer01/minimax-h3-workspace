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


def resolution(aspect, quality, form=None):
    """Size from the format + quality, or the user's own width/height (rounded to a multiple of 32)."""
    if form and str(form.get("width", "")).isdigit() and str(form.get("height", "")).isdigit():
        cw, ch = int(form["width"]), int(form["height"])
        if cw >= 64 and ch >= 64:
            return min(2048, r32(cw)), min(2048, r32(ch))
    w, h = ASPECTS[aspect]
    px = QUALITY[quality]
    return r32(math.sqrt(px * w / h)), r32(math.sqrt(px * h / w))


def frames(seconds):
    # same formula as the workflow's ComfyMathExpression: 24 fps, length = 17k + 5
    f = max(5, round(seconds * FPS))
    return f + (5 - f % 17) % 17


def video_graph(m, prompt, width, height, seconds, seed, first=None, last=None, extra_loras=(), fast=None):
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
    loras = ([(fast, 1.0)] if fast else []) + list(extra_loras)  # Turbo-SLA first, then the user's own LoRAs
    if loras:
        g["2"]["inputs"]["model"] = _lora_chain(g, ["1", 0], loras)
    if fast:  # "fast" switch: Turbo-SLA 4 steps (flow shifts 6 / 3 and 15% sparse attention as in its own setup)
        g["2"]["inputs"].update(shift_video=6, shift_audio=3)
        g["4"]["inputs"]["selection.keep_percent"] = 15.0
        g["11"]["inputs"]["steps"] = 4

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
              ref_size="max", extra_loras=(), pika=None):
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
    if pika:  # Pika H3 Flex Turbo (8 steps, HD): its own LoRA instead of the 4-step one; no sigma shift, scheduler "simple"
        mref = _lora_chain(g, ["2", 0], list(extra_loras) + [(pika, 1.0)])
        steps, scheduler = max(6, steps), "simple"
    else:
        model = _lora_chain(g, ["2", 0], list(extra_loras) + [(m["ref2v_turbo_lora"], 0.7)])
        g["9"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 4.0}}
        mref, scheduler = ["9", 0], "beta"
    g.update({
        "10": {"class_type": "BasicScheduler", "inputs": {"model": mref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": mref, "conditioning": ["8", 0]}},
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


# MiniMax_H3_Direct-ai-Actor: one pass, 10 steps with the author's own sigma list
ACTOR_SIGMAS = "1, .9882, .973, .9524, .9231, .878, .8, .6316, .4737, .1579, 0"


def actor_graph(m, prompt, width, height, seconds, seed, images, audios, realism=None, extra_loras=()):
    """Acting scenes: first frame + a voice sample, new speech written in the prompt (<d>...</d>).

    Same Singularity Ref2V model as the references mode, but with sol-attn sparse attention,
    sigma shift 12.2 / 3, er_sde and the turbo LoRA at full strength, as in the original workflow.
    `realism` is the (file, strength) of the people-realism LoRA when it is on.
    """
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["lipsync_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["1", 0], "attention": "comfy kitchen attention"}},
        "3": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["2", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.2, "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
            "verbose": False}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["text_encoder"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["video_vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.2, "shift_audio": 3.0}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0], "audio_vae": ["6", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames(seconds), "ref_image_size": "match"}},
    }
    loras = [(m["ref2v_turbo_lora"], 1.0)] + ([realism] if realism else []) + list(extra_loras)
    model = _lora_chain(g, ["7", 0], loras)
    g.update({
        "10": {"class_type": "ManualSigmas", "inputs": {"sigmas": ACTOR_SIGMAS}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["8", 0]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["13", 0], "sampler": ["12", 0], "sigmas": ["10", 0], "latent_image": ["8", 1]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 1], "vae": ["5", 0]}},
        "16": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 1], "vae": ["6", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {
            "images": ["15", 0], "audio": ["16", 0], "fps": float(FPS), "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}},
        "18": {"class_type": "SaveVideo", "inputs": {
            "video": ["17", 0], "filename_prefix": "H3Workspace/actor", "format": "auto", "format.codec": "auto"}},
    })
    loaders = _refs(g, "8", images, (), audios)
    stages = {n: "load" for n in ["1", "2", "3", "4", "5", "6", "7"] + [k for k in g if k.isdigit() and 60 <= int(k) < 70]}
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


# ---------- pictures: Krea 2 (text) and Qwen Image 2.1 (edit, character sheet) ----------

# Krea 2 works at about 1 MP only
KREA_SIZES = {"16:9": (1376, 768), "9:16": (768, 1376), "1:1": (1024, 1024), "4:3": (1184, 896),
              "3:4": (896, 1184), "3:2": (1248, 832), "2:3": (832, 1248), "21:9": (1568, 672)}
QWEN_MP = {"1": 1.0, "2": 2.0}
SHEET_SIZE = (1920, 1088)
SHEET6_SIZE = (2112, 960)  # six tall panels in one row, about 2 MP like the four-panel sheet
SHEET_BG = {"grey": "light grey", "white": "white", "black": "black"}


def krea_graph(m, prompt, width, height, seed):
    """Krea 2 Turbo, text to picture: 10 steps, CFG 1, no negative prompt (the model has none)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["krea_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 10, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea", "images": ["8", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "encode", "5": "encode", "6": "encode",
              "7": "sample", "8": "decode", "9": "save"}
    return g, stages


KREA_REAL_SCALES = {"2k": 1.5, "4k": 2.5}


def krea_real_graph(m, prompt, width, height, seed, scale):
    """Krea 2 realism (RedCraft RedMix + Realism Engine LoRA 0.5), two passes: draw ~1 MP, enlarge the latent, refine with denoise 0.6."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["kreareal_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["krea_clip"], "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["krea_vae"]}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": m["kreareal_lora"], "strength_model": 0.5, "model": ["1", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 8, "cfg": 1, "sampler_name": "euler_ancestral", "scheduler": "sgm_uniform", "denoise": 1,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "LatentUpscaleBy", "inputs": {"upscale_method": "bicubic", "scale_by": scale, "samples": ["8", 0]}},
        "10": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 6, "cfg": 1, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 0.6,
            "model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["9", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/krea_real", "images": ["11", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "sample", "9": "upscale", "10": "sample2", "11": "decode", "12": "save"}
    return g, stages


def _qwen_base(m, image, megapixels, prompt, seed, size=None):
    """The graph of the "One Image Edit" / "Character Sheet" workflows on plain ComfyUI nodes.
    `size` (width, height) gives an empty canvas of its own; otherwise the result keeps the picture's shape."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 32, "image": ["5", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0],
            "images.image_1": ["6", 0], "vae": ["3", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["7", 2]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen", "images": ["10", 0]}},
    }
    if size:
        g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
        g["9"]["inputs"]["latent_image"] = ["8", 0]
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "5": "encode", "6": "encode", "7": "encode",
              "8": "encode", "9": "sample", "10": "decode", "11": "save"}
    return g, stages


def flux_t2i_graph(m, prompt, width, height, seed):
    """FLUX.2 klein 4B (distilled), picture by text: 4 steps, CFG 1, no negative prompt."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0], "guider": ["6", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["7", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_t2i", "images": ["12", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "encode", "5": "encode", "6": "encode", "7": "encode",
              "8": "encode", "9": "encode", "10": "encode", "11": "sample", "12": "decode", "13": "save"}
    return g, stages


def qwen_t2i_graph(m, prompt, negative, width, height, seed):
    """Qwen Image 2.1, picture by text (no reference). With a negative prompt CFG goes above 1 so that it works."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "7": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "prompt": prompt, "negative_prompt": negative, "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 25, "cfg": 3 if negative.strip() else 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_t2i", "images": ["10", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "save"}
    return g, stages


STORY_SIZE = (1920, 1088)


def qwen_story_graph(m, images, prompt, seed, size=STORY_SIZE):
    """Storyboard from 1-3 pictures (character, location...) on Qwen Image 2.1: <image1>, <image2> in the prompt."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
    }
    enc = {"prompt": prompt, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
    stages = {"1": "load", "2": "load", "3": "load", "4": "load"}
    for i, img in enumerate(images, 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        enc[f"images.image_{i}"] = [b, 0]
        stages[a] = stages[b] = "encode"
    g["7"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    g["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["9"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
        "model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1], "latent_image": ["8", 0]}}
    g["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}}
    g["11"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/qwen_story", "images": ["10", 0]}}
    stages.update({"7": "encode", "8": "encode", "9": "sample", "10": "decode", "11": "save"})
    return g, stages


FRAME_COLS = {2: 2, 3: 3, 4: 2, 6: 3, 8: 4, 9: 3}  # frames -> columns of the sheet (rows follow)
FRAME_SIZE = (1376, 768)
FRAME_LINE = {"white": ("white", 6), "black": ("black", 4), "none": ("white", 0)}


def _glue(g, stages, titles, frame_nodes, line):
    """Glue the decoded frames into one sheet with ImageStitch (exact thin line), saved as "Лист раскадровки"."""
    n = len(frame_nodes)
    color, gap = FRAME_LINE.get(line, FRAME_LINE["white"])
    cols = FRAME_COLS.get(n, n)
    rows = [frame_nodes[k:k + cols] for k in range(0, n, cols)]
    row_out = []
    for r, items in enumerate(rows):
        cur = items[0]
        for k, nxt in enumerate(items[1:]):
            nid = f"4{r}{k}"
            g[nid] = {"class_type": "ImageStitch", "inputs": {
                "image1": cur, "direction": "right", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
            stages[nid] = "decode"
            cur = [nid, 0]
        row_out.append(cur)
    cur = row_out[0]
    for r, nxt in enumerate(row_out[1:]):
        nid = f"49{r}"
        g[nid] = {"class_type": "ImageStitch", "inputs": {
            "image1": cur, "direction": "down", "match_image_size": True, "spacing_width": gap, "spacing_color": color, "image2": nxt}}
        stages[nid] = "decode"
        cur = [nid, 0]
    g["500"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_sheet", "images": cur}}
    stages["500"] = "save"
    titles["500"] = "Лист раскадровки"


def flux_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white"):
    """Storyboard frame by frame on FLUX.2 klein: every frame is its own picture (saved separately, full size) and the
    sheet is glued from them by ImageStitch with an exact thin line, so there are no margins or double frames."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "13": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "13": "encode", "14": "encode", "15": "encode", "8": "encode"}
    lat = []
    for i, img in enumerate(refs[:3], 1):
        a, b, v = f"10{i}", f"11{i}", f"12{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        lat.append(v)
        for k in (a, b, v):
            stages[k] = "encode"
    g["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
    neg = ["9", 0]
    for j, v in enumerate(lat):  # the negative side does not depend on the frame: build it once
        nid = f"15{j}"
        g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        neg = [nid, 0]
        stages[nid] = "encode"
    stages["9"] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        t = str(base)
        g[t] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}}
        pos = [t, 0]
        for j, v in enumerate(lat):
            nid = str(base + 1 + j)
            g[nid] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
            pos = [nid, 0]
            stages[nid] = "encode"
        gd, nz, sm, dc, sv = (str(base + k) for k in (5, 6, 7, 8, 9))
        g[gd] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}}
        g[nz] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + i}}
        g[sm] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [nz, 0], "guider": [gd, 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [sm, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({t: "encode", gd: "encode", nz: "encode", sm: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def qwen_frames_graph(m, refs, prompts, seed, size=FRAME_SIZE, line="white"):
    """The same frame-by-frame storyboard on Qwen Image 2.1 (slower than FLUX, no commercial license)."""
    n = len(prompts)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["qwen_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["qwen_clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["qwen_vae"]}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "4": "load", "8": "encode"}
    scaled = []
    for i, img in enumerate(refs[:3], 1):
        a, b = f"10{i}", f"11{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32, "image": [a, 0]}}
        scaled.append(b)
        stages[a] = stages[b] = "encode"
    titles, frame_nodes = {}, []
    for i, text in enumerate(prompts):
        base = 200 + i * 10
        en, ks, dc, sv = (str(base + k) for k in (0, 1, 2, 3))
        enc = {"prompt": text, "negative_prompt": "", "resolution": 0, "clip": ["2", 0], "vae": ["3", 0]}
        for k, b in enumerate(scaled, 1):
            enc[f"images.image_{k}"] = [b, 0]
        g[en] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
        g[ks] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": 25, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["4", 0], "positive": [en, 0], "negative": [en, 1], "latent_image": ["8", 0]}}
        g[dc] = {"class_type": "VAEDecode", "inputs": {"samples": [ks, 0], "vae": ["3", 0]}}
        g[sv] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/story_frame", "images": [dc, 0]}}
        stages.update({en: "encode", ks: "sample", dc: "decode", sv: "save"})
        titles[sv] = f"Кадр {i + 1}"
        frame_nodes.append([dc, 0])
    _glue(g, stages, titles, frame_nodes, line)
    return g, stages, titles


def flux_story_graph(m, images, prompt, seed, size=STORY_SIZE):
    """Storyboard from 1-3 pictures on FLUX.2 klein: every picture is one more ReferenceLatent (image 1, image 2 ...)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
    }
    stages = {"1": "load", "2": "load", "3": "load", "7": "encode", "8": "encode"}
    pos, neg = ["7", 0], ["8", 0]
    for i, img in enumerate(images, 1):
        a, b, v, rp, rn = f"10{i}", f"11{i}", f"12{i}", f"13{i}", f"14{i}"
        g[a] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[b] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16, "image": [a, 0]}}
        g[v] = {"class_type": "VAEEncode", "inputs": {"pixels": [b, 0], "vae": ["3", 0]}}
        g[rp] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [v, 0]}}
        g[rn] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [v, 0]}}
        pos, neg = [rp, 0], [rn, 0]
        for k in (a, b, v, rp, rn):
            stages[k] = "encode"
    g["12"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": pos, "negative": neg, "cfg": 1}}
    g["13"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}}
    g["14"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": size[0], "height": size[1]}}
    g["15"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["16"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["17"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["16", 0], "guider": ["12", 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}}
    g["18"] = {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}}
    g["19"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux_story", "images": ["18", 0]}}
    stages.update({"12": "encode", "13": "encode", "14": "encode", "15": "encode", "16": "encode",
                   "17": "sample", "18": "decode", "19": "save"})
    return g, stages


def flux_edit_graph(m, prompt, image, megapixels, seed):
    """FLUX.2 klein 4B (distilled), edit a picture: the official ComfyUI template on plain nodes, 4 steps, CFG 1.
    The result keeps the shape of the picture (sides in steps of 16)."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["flux_unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["flux_clip"], "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["flux_vae"]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16, "image": ["4", 0]}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "9": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["9", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["9", 0]}},
        "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["11", 0], "cfg": 1}},
        "13": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["6", 0], "height": ["6", 1], "batch_size": 1}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": ["6", 0], "height": ["6", 1]}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["16", 0], "guider": ["12", 0], "sampler": ["15", 0], "sigmas": ["14", 0], "latent_image": ["13", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/flux", "images": ["18", 0]}},
    }
    stages = {str(i): "load" for i in (1, 2, 3)}
    stages.update({str(i): "encode" for i in range(4, 17)})
    stages.update({"17": "sample", "18": "decode", "19": "save"})
    return g, stages


def qwen_edit_size(aspect, megapixels):
    """Canvas for a chosen aspect ratio (about 1 or 2 megapixels, sides in steps of 32); None = keep the picture's own shape."""
    if aspect not in KREA_SIZES:
        return None
    w, h = KREA_SIZES[aspect]
    k = (megapixels or 1.0) ** 0.5
    return (round(w * k / 32) * 32, round(h * k / 32) * 32)


def qwen_edit_graph(m, prompt, image, megapixels, seed, size=None):
    return _qwen_base(m, image, megapixels, prompt, seed, size)


def sheet_prompt(headless=True, background="grey", marks="", panels=4):
    """Four panels: full body front (without the head by default), full body back, close-up front, close-up profile.
    Tested on Qwen Image 2.1 with a cartoon character: same style and outfit on all panels."""
    bg = SHEET_BG.get(background, "light grey")
    first = ("full-body front view, the figure has no head at all, not a mannequin and not a neck stump, the collar is "
             "simply empty and the plain background shows through the neckline, standing in a relaxed neutral pose, "
             "showing the whole outfit from the neckline to the shoes." if headless else
             "full-body front view facing the viewer, standing in a relaxed neutral pose, showing the whole outfit "
             "from head to shoes.")
    keep = "Keep the same face, hairstyle, " + (marks.strip().rstrip(".,") + ", " if marks.strip() else "") \
        + "clothing, colors, proportions and art style as in <image1> in every panel."
    if panels == 6:
        return (
            "A character reference sheet of the character from <image1>, split into six equal tall panels side by side in one "
            "row, all at the same scale, in the same art style as <image1>.\n"
            f"Panel one: {first}\n"
            "Panel two: full-body three-quarter view of the same character, turned about forty-five degrees to the left, "
            "same pose and outfit, head fully visible.\n"
            "Panel three: full-body side view in profile facing left, same pose and outfit, head fully visible.\n"
            "Panel four: full-body back view of the same character, same pose and outfit, seen from behind, head fully visible.\n"
            "Panel five: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
            "Panel six: close-up head-and-shoulders portrait in three-quarter view, with the same lighting and scale as panel five.\n"
            f"{keep}\n"
            f"Plain solid {bg} background, soft even lighting, no text, no labels.")
    return (
        "A character reference sheet of the character from <image1>, split into four equal panels side by side in one "
        "row, all at the same scale, in the same art style as <image1>.\n"
        f"Panel one: {first}\n"
        "Panel two: full-body back view of the same character in the same pose and outfit, seen from behind, head fully visible.\n"
        "Panel three: close-up head-and-shoulders portrait facing the viewer with a neutral expression.\n"
        "Panel four: close-up head-and-shoulders portrait in profile, with the same lighting and scale as panel three.\n"
        f"{keep}\n"
        f"Plain solid {bg} background, soft even lighting, no text, no labels.")


def qwen_sheet_graph(m, image, seed, headless=True, background="grey", marks="", panels=4):
    prompt = sheet_prompt(headless, background, marks, panels)
    size = SHEET6_SIZE if panels == 6 else SHEET_SIZE
    g, stages = _qwen_base(m, image, size[0] * size[1] / 1048576, prompt, seed, size=size)
    return g, stages, prompt


# ---------- improve quality of a finished video: SeedVR2 on plain ComfyUI nodes, one step, tiled ----------

UPSCALE_SCALES = {"1.5": 1.5, "2": 2.0}
UPSCALE_SEC_PER_MP_FRAME = 3.24  # measured on an RTX 4060 Ti 16 GB: 5 s of 1056x608 -> 1584x912 took ~10 minutes


def upscale_image_graph(m, image, scale, seed):
    """The same SeedVR2 one-step enlarge for a single picture."""
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["1", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "H3Workspace/upscale", "images": ["11", 0]}},
    }
    stages = {"1": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save"}
    return g, stages


def upscale_graph(m, video, scale, seed):
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"upscale_method": "lanczos", "scale_by": scale, "image": ["2", 0]}},
        "4": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["3", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["seedvr_vae"]}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": m["seedvr_unet"], "weight_dtype": "default"}},
        "7": {"class_type": "VAEEncodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "pixels": ["4", 0], "vae": ["5", 0]}},
        "8": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["6", 0], "vae_conditioning": ["7", 0]}},
        "9": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1,
            "model": ["6", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecodeTiled", "inputs": {
            "tile_size": 512, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8, "samples": ["9", 0], "vae": ["5", 0]}},
        "11": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "color_correction_method": "lab", "images": ["10", 0], "original_resized_images": ["3", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "13": {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0], "filename_prefix": "H3Workspace/upscale", "format": "auto", "format.codec": "auto"}},
    }
    stages = {"1": "load", "2": "load", "5": "load", "6": "load", "3": "encode", "4": "encode", "7": "encode", "8": "encode",
              "9": "sample", "10": "decode", "11": "decode", "12": "save", "13": "save"}
    return g, stages

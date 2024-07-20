from diffusers import DiffusionPipeline, AutoencoderKL, StableDiffusionPipeline, DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, DPMSolverSinglestepScheduler, KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler, DDIMInverseScheduler, DEISMultistepScheduler, IPNDMScheduler, KarrasVeScheduler, ScoreSdeVeScheduler, LCMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from dramatiq import actor
import torch
import gc
from configs.config import PROJECT_FOLDER
import os
from utils.cloudinary import write_image_to_cloudinary
from typing import Optional

def return_scheduler(scheduler: str):
    schedulers = {
        "DDPMScheduler": DDPMScheduler,
        "DDIMScheduler": DDIMScheduler,
        "PNDMScheduler": PNDMScheduler,
        "LMSDiscreteScheduler": LMSDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
        "HeunDiscreteScheduler": HeunDiscreteScheduler,
        "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
        "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
        "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "DDIMInverseScheduler": DDIMInverseScheduler,
        "DEISMultistepScheduler": DEISMultistepScheduler,
        "IPNDMScheduler": IPNDMScheduler,
        "KarrasVeScheduler": KarrasVeScheduler,
        "ScoreSdeVeScheduler": ScoreSdeVeScheduler,
        "LCMScheduler": LCMScheduler
    }
    return schedulers.get(scheduler)

def get_inputs(batch_size: int, prompt: str, height: int, width: int, safety_checker: bool, guidance_scale: int, negative_prompt: Optional[str] = "", seed: Optional[int] = None, scheduler: Optional[str] = "", loras: Optional[str] = "", inference_steps: int = 8):
    params = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "num_images_per_prompt": max(batch_size, 1),
        "num_inference_steps": inference_steps * batch_size  or 50
    }
    if seed:
        params['seed'] = seed
    if safety_checker:
        safety_checker = StableDiffusionSafetyChecker()
        params['safety_checker'] = safety_checker
    if loras:
        unet = UNet2DConditionModel.from_pretrained(loras, torch_dtype=torch.float16, variant="fp16")
        params['unet'] = unet
    if scheduler:
        scheduler_instance = return_scheduler(scheduler)
        if scheduler_instance:
            params['scheduler'] = scheduler_instance
        else:
            print(f"Scheduler {scheduler} not found. Using default scheduler.")

    return params

def validate_and_return_inputs(dto):
    try:
        inputs = get_inputs(
            batch_size=dto['samples'],
            prompt=dto['text'],
            negative_prompt=dto['negative_prompt'],
            height=dto["height"],
            width=dto["width"],
            safety_checker=dto["safety_checker"],
            guidance_scale=dto["guidance_scale"],
            seed=dto["seed"],
            scheduler=dto["scheduler"],
            loras=dto["loras"],
            inference_steps=dto["num_inference_steps"]
        )
        print("Inputs: %s", inputs)

        if not inputs:
            raise ValueError("Inputs are not properly initialized.")

        # Check for NoneType in inputs
        for key, value in inputs.items():
            if value is None:
                raise ValueError(f"Input parameter {key} is None")
        return inputs

    except Exception as e:
        print(f"Error generating image: {e}")
        return str(e)

def flush_memory():
    """Frees up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_pipeline_and_unload_pipeline(model_name, vae=None, loras=None, unload_memory=True):
    save_path = f"{PROJECT_FOLDER}/code/pipelines/{model_name.replace('/', '-')}.pt"
    if not os.path.exists(save_path):

        pipeline = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        torch.save(pipeline, save_path)
        print(f"Saved {model_name} pipeline to {save_path}")
        if unload_memory:
            unload_pipeline(pipeline)
    else:
        print("Pipeline already saved")

# @actor
def unload_pipeline(pipeline):
    del pipeline
    pipeline = None
    flush_memory()
    print(f"Unloaded pipeline from memory.")

# @actor
def load_pipeline(model_name):
    try:
        save_path = f"{PROJECT_FOLDER}/code/pipelines/{model_name.replace('/', '-')}.pt"
        if os.path.exists(save_path):
            # pipeline = torch.load(save_path, map_location=torch.device("cuda"), mmap=True)
            pipeline = torch.load(save_path)
            pipeline.unet.config.addition_embed_type = None 
            print("Pipeline loaded successfully")
            return pipeline
        else:
            print("Pipeline failed to load")
            return None
    except Exception as e:
        print(f"Pipeline failed to load. Detail: {str(e)}")

def make_image_from_prompt(pipeline, dto, image=None):
    """
    Generate an image from a text prompt using the given pipeline.

    :param pipeline: A loaded DiffusionPipeline instance.
    :param prompt: Textual prompt describing the image to generate.
    :param num_inference_steps: Number of inference steps to perform. Default is 1000.
    :return: The generated image.
    """
    if image is not None:
        result = pipeline(prompt=dto['text'], num_inference_steps=dto['num_inference_steps'], denoising_start=0.8, image=image)
        print("Refiner pipeline")
        image = result.images[0]  # Assuming the result contains a list of images
    else:
        inputs = validate_and_return_inputs(dto)
        print(inputs)
        result = pipeline(**inputs)
        print("Base pipeline")
        image = result.images
    return image


# get estimate of time required for task
def time_estimate(params):
    area = int(params["width"]) * int(params["height"])
    steps_50_time = 3.2e-8 * area**1.5
    if area < 250000:
        steps_50_time = 4
    return steps_50_time * (int(params["num_inference_steps"]) / 50) * int(params["samples"])

def custom_embeddings(type: str, pipeline):
    if type == 'sdxl':
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
        pipeline.vae = vae
        pipeline.unet = unet
    elif type == 'sd1.5':
        unet_id = "mhdang/dpo-sd1.5-text2image-v1"
        unet = UNet2DConditionModel.from_pretrained(
            unet_id,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler=DDPMScheduler()
    elif type == 'sd1':
        pipeline.scheduler=DDPMScheduler()
    return pipeline
        

@actor
def image_generation(dto, uuids, enable_custom_embeddings = True):
    print(f"Image generation process started for {dto['base_model']} {dto['id']}")
    save_pipeline_and_unload_pipeline(dto['id'])
    loaded_base = load_pipeline(dto["id"])
    if not loaded_base:
        raise Exception("Base pipeline loading error")
    # if enable_custom_embeddings:
    #     loaded_base = custom_embeddings(dto['base_model'], loaded_base)
    loaded_base = loaded_base.to("cuda")
    images = make_image_from_prompt(pipeline=loaded_base, dto=dto)
    unload_pipeline(loaded_base)
    # loaded_refiner = load_pipeline("stabilityai/stable-diffusion-xl-refiner-1.0")
    # if not loaded_refiner:
    #     raise Exception("Refiner pipeline loading error")
    # image = make_image_from_prompt(pipeline=loaded_refiner, prompt=prompt, num_inference_steps=20, image=images)
    for uuid, image in zip(uuids, images):
        write_image_to_cloudinary(image, uuid)
        image.save("output.png")
        print(f"Image uploaded for {uuid}")

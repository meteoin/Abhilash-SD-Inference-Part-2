
import os
from dramatiq import Middleware
from configs.config import PROJECT_FOLDER, IMAGE_URL, redis_broker
import os
import uuid
from dto.sd_types import TextToImageInput
import json
from fastapi import FastAPI
from tasks import save_pipeline_and_unload_pipeline, image_generation, time_estimate

class BeforeWorkerBootMiddleware(Middleware):
    def before_worker_boot(self, broker, worker):
        # Code to execute before the worker process starts
        print("Worker is about to start")
        on_worker_init()
        print("Worker init done")


middleware = BeforeWorkerBootMiddleware()
redis_broker.add_middleware(middleware)
app = FastAPI()

def on_worker_init():
    model_names = []
    json_file_path = f'{PROJECT_FOLDER}/code/models.json'
    
    # Load model names from the JSON file
    with open(json_file_path, 'r') as file:
        model_names = json.load(file)
    print(model_names)
    for model_name in model_names:
        print(model_name)
        save_path = f"{PROJECT_FOLDER}/code/pipelines/{model_name.replace('/', '-')}.pt"
        if not os.path.exists(save_path):
            save_pipeline_and_unload_pipeline(model_name)
    return "All pipelines saved and loaded"
# @actor
@app.post("/invocation")
def image_generation_pipeline(dto: TextToImageInput):
    # Optionally, load the pipelines when needed
    # loaded_refiner = load_pipeline.send("stabilityai/stable-diffusion-xl-refiner-1.0")
    # Load the pt files and make inference
    # on_worker_init()
    # Save this on worker init as pkt
    # base = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir="/tmp/"
    # ).to("cuda")

    # dto = TextToImageInput(
    #   text= "A majestic lion jumping from a big stone at night",
    #   id="stabilityai/stable-diffusion-xl-base-1.0",
    #   base_model= "sdxl",
    #   samples= 1,
    #   width= 1024,
    #   height= 1024,
    #   guidance_scale= 1,
    #   safety_checker= False,
    #   num_inference_steps= 20,
    #   negative_prompt= "",
    # #   seed=1,
    #   scheduler= "DDIMScheduler",
    #   loras= None
    # )
    uuids = []
    urls = []
    for i in range(dto.samples):
        uuid_generated_name = str(uuid.uuid4())
        uuids.append(uuid_generated_name)
        urls.append(f"{IMAGE_URL}/{uuid_generated_name}.png")
        print(f"Image URL: {IMAGE_URL}/{uuid_generated_name}.png")
    dto = dto.dict()
    image_generation.send(dto, uuids)
    time = time_estimate(dto)
    print(f"Image generation task completed. ETA for image appearance {time}")
    return {"ETA": time, "URLs": urls}

@app.get('/')
def main():
    return "Initalized API"
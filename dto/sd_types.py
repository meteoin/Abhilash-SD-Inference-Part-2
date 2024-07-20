from pydantic import BaseModel
from typing import Optional

class TextToImageInput(BaseModel):
    text: str
    id:str
    base_model:str
    samples: int
    width: int
    height: int
    guidance_scale: float
    safety_checker: bool
    num_inference_steps: Optional[int] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    loras: Optional[str] = None

class upload_Image(BaseModel):
    id:str
    

class ImageToImageInput(BaseModel):
    input_image_url: str

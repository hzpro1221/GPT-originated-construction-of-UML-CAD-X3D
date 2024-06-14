import torch
import torch.nn as nn

from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image
 
class StableDiffusionv1point5(nn.Module):
    def __init__(self):
        super().__init__()
        # Pipeline for text -> image
        self.pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
	        "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16, 
            variant="fp16").to("cuda")

        # Pipeline for text + image -> image
        self.pipeline_img2img = AutoPipelineForImage2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True)
        self.pipeline_img2img.enable_model_cpu_offload()
        self.pipeline_img2img.enable_xformers_memory_efficient_attention()
    
    def generate(self, lm_output, addition_content, image_path):
        if (addition_content != 'None'):
            prompt = lm_output + addition_content
        else:
            prompt = lm_output

        if (image_path != 'None'):
            init_image = load_image(image_path)
            image = self.pipeline_img2img(prompt, image=init_image).images[0]
            return image
        else:
            generator = torch.Generator("cuda").manual_seed(31)
            image = self.pipeline_text2img(prompt, generator=generator).images[0]
            return image
import torch
from diffusers import StableDiffusionPipeline, LCMScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    torch_dtype=torch.float16, 
    safety_checker=None
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()

print(f"Diffusers type: {type(pipe.scheduler)}")
try:
    img = pipe("a dog", num_inference_steps=6, guidance_scale=1.5).images[0]
    print("Success without fixing!")
except Exception as e:
    print(f"Failed with 6 steps: {e}")

try:
    img = pipe("a dog", num_inference_steps=4, guidance_scale=1.5).images[0]
    print("Success with 4 steps!")
except Exception as e:
    print(f"Failed with 4 steps: {e}")

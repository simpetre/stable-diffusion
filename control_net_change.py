import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import safetensors
import requests
import os

# Ensure you're using a GPU if available (recommended for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Set the scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load Loras from Hugging Face Hub
lora_paths = {
    "detail_slider_v4": "https://huggingface.co/lraza8632/LoRAs/resolve/main/detail_slider_v4.safetensors",
    "more_details": "https://huggingface.co/lraza8632/LoRAs/resolve/main/more_details.safetensors",
}

for name, url in lora_paths.items():
    # Download the Lora weights
    response = requests.get(url)
    
    # Save the downloaded Lora weights to a temporary file
    lora_path = f"{name}.safetensors"
    with open(lora_path, "wb") as f:
        f.write(response.content)
    
    # Load the Lora weights
    lora_weights = safetensors.torch.load_file(lora_path, device=device)
    
    # Apply the Lora weights to the UNet model
    pipe.unet.load_state_dict(lora_weights, strict=False)
    
    # Remove the temporary file
    os.remove(lora_path)

# Your prompt
prompt = "a realistic anthropomorphic hedgehog in a painted gold robe, standing over a bubbling cauldron, an alchemical circle, steam and haze flowing from the cauldron to the floor, glow from the cauldron, electrical discharges on the floor, Gothic, bokeh, depth of field, blurry background, shallow focus, <lora:detail_slider_v4:2>, <lora:more_details:1.0>"

# Negative prompt
negative_prompt = "bad_pictures, easynegative, ng_deepnegative_v1_75t, Unspeakable-Horrors-64v, kkw-new-neg-v1.6"

# Generate an image
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=5,
        temperature=1,
        width=768,
        height=512,
    ).images[0]

# Save the image
output.save("generated_image.png")
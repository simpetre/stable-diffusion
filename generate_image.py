import torch
from diffusers import StableDiffusionPipeline

# Ensure you're using a GPU if available (recommended for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Your prompt
prompt = "A picture of a mouse eating a bagel"

# Generate an image
with torch.no_grad():
    output = pipe(prompt="Two women rowing along the river Thames",
              num_inference_steps=400,
              guidance_scale=15,
              temperature=0.5,
              ).images[0]

# Save the image
output.save("generated_image.png")

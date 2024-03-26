import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import HEDdetector

# Ensure you're using a GPU if available (recommended for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load the ControlNet model
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
    print("ControlNet model loaded successfully.")
except Exception as e:
    print("An error occurred while loading the ControlNet model:", e)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to(device)

# Set the scheduler (UniPCMultistepScheduler is recommended for ControlNet)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Load and prepare your image
image = load_image("generated_image.png")
detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
edge_image = detector(image)

# Your prompt
prompt = "A pirate holding a blunderbuss"

# Generate an image
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        image=edge_image,
        num_inference_steps=100,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
    ).images[0]

# Save the image
output.save("generated_image_controlnet.png")
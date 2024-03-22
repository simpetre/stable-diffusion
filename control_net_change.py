import torch
from controlnet_aux import HEDdetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Ensure you're using a GPU if available (recommended for performance)
device = "cuda" if torch.cuda.is_available() else "cpu"

# The following code assumes that the diffusers and transformers libraries are up-to-date
# and that the 'lllyasviel/sd-controlnet-canny' model exists in the Hugging Face repository.

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

# Prepare your image
image = "generated_image.png"
detector = HEDdetector.from_pretrained('lllyasviel/sd-controlnet-canny')
edge_image = detector(image)

# Your prompt
prompt = "A futuristic city skyline at sunset, in the style of Vincent van Gogh"

# Generate an image
with torch.no_grad():
    output = pipe(
        prompt=prompt,
        image=edge_image,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
    ).images[0]

# Save the image
output.save("generated_image_controlnet.png")
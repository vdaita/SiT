from flux_client import *

# Assume `pipe` is an instance of Flux2KleinPipeline
prompt = "A futuristic city at sunset"
cond_img = PIL.Image.open("reference.png")

# 1️⃣  Prepare everything (you can reuse the returned tensors later)
(
    prompt_embeds,
    text_ids,
    latents,
    latent_ids,
    image_latents,
    image_latent_ids,
    timesteps,
    _,
) = pipe.prepare_input(
    prompt=prompt,
    image=cond_img,
    height=1024,
    width=1024,
    num_images_per_prompt=1,
    generator=torch.Generator().manual_seed(42),
    guidance_scale=4.0,
    max_sequence_length=512,
    text_encoder_out_layers=(9, 18, 27),
)

# 2️⃣  Run the diffusion loop (you could modify `guidance_scale` on‑the‑fly)
final_latents = pipe.forward_pass(
    prompt_embeds=prompt_embeds,
    text_ids=text_ids,
    latents=latents,
    latent_ids=latent_ids,
    timesteps=timesteps,
    image_latents=image_latents,
    image_latent_ids=image_latent_ids,
    guidance_scale=4.0,
)

# 3️⃣  Decode (and automatically save to `outputs/`)
image = pipe.decode_output(final_latents, output_type="pil")
image.show()
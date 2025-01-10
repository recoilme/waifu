# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
import torch
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "/home/recoilme/models/wtst",
    variant="fp16",
    #torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Load the text encoders and tokenizers.
text_encoder = CLIPTextModel.from_pretrained(pipe_id, subfolder="text_encoder", torch_dtype=torch.float16,variant="fp16").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(pipe_id, subfolder="tokenizer")


pipe.vae.to(torch.float16)
pipe.text_encoder.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
image = pipe(
    prompt=prompt,
    height=512,
    width=768,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
)[0]

image[0].save("sana.png")
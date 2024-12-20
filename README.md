<p align="center" style="border-radius: 10px">
  <img src="asset/logo_waifu.png" width="35%" alt="logo"/>
</p>

# ⚡️Waifu: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer 


<p align="center" border-raduis="10px">
  <img src="asset/waifu.jpg" width="90%" alt="teaser_page1"/>
</p>

## 💡 Introduction

tldr; We just need a model to generate waifu


We introduce Waifu, a text-to-image framework that can efficiently generate images up to 768 × 768 resolution on 80+ languages. Our goal was to create a small model that is easy to use and full-tune on custom GPU, but without compromising on quality. It's like a SD 1.5, but developed in 2024 using the most advanced components at this moment. Waifu can synthesize high-resolution, high-quality images of waifu with strong text-image alignment at a remarkably fast speed, deployable on laptop GPU.


Core designs include:

(1) [**AuraDiffusion/16ch-vae**](https://huggingface.co/AuraDiffusion/16ch-vae): A fully open source 16ch VAE. Natively trained in fp16. \
(2) **Linear DiT**: we replace all vanilla attention in DiT with linear attention, which is more efficient at high resolutions without sacrificing quality. \
(3) [**MEXMA-SigLIP**](https://huggingface.co/visheratin/mexma-siglip): MEXMA-SigLIP is a model that combines the [MEXMA](https://huggingface.co/facebook/MEXMA) multilingual text encoder and an image encoder from the [SigLIP](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) model. This allows us to get a high-performance CLIP model for 80 languages.. \
(4) **Efficient training and sampling**: we use **Flow-Eulerr** as sampler, adapt Adafactor-Fused optimizer and use bf16 precision for training, with efficient caption labeling and selection to accelerate convergence.

As a result, Waifu-2b is very competitive with modern giant diffusion model (e.g. Flux-12B), being 20 times smaller and 100+ times faster in measured throughput. Moreover, Waifu-2b can be deployed on a 16GB laptop GPU, taking less than 1 second to generate a 768 × 768 resolution image. Waifu enables waifu creation at low cost.


 // AiArtLab team
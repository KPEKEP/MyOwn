import argparse
import logging

import torch
from PIL import Image
from transformers import CLIPTokenizer
import modeling.sd.model_loader as model_loader
from modeling.sd.pipeline import generate
from time import sleep

def main():
    parser = argparse.ArgumentParser(description="Generate images based on given parameters.")

    parser.add_argument("--device", default="cpu", help="Device to run the models on.")
    parser.add_argument("--vocab_file", default="data/vocab.json", help="Path to the vocabulary file for the tokenizer.")
    parser.add_argument("--merges_file", default="data/merges.txt", help="Path to the merges file for the tokenizer.")
    parser.add_argument("--model_file", default="data/v1-5-pruned-emaonly.ckpt", help="Path to the model checkpoint file.")
    parser.add_argument("--prompt", default="air baloon, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.", help="Text prompt for conditional image generation.")
    parser.add_argument("--uncond_prompt", default="blurred, unclear, distant", help="Text prompt for unconditional image generation.")
    parser.add_argument("--do_cfg", default=True, type=bool, help="Whether to perform conditional image generation.")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Scale factor for conditional image generation.")
    parser.add_argument("--input_image_path", default=None, help="Path to an optional input image for conditional image generation.")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the noise added to the latents.")
    parser.add_argument("--sampler_name", default="ddpm", help="Name of the sampler to use.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for the sampler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Check for CUDA availability
    if torch.cuda.is_available():
        args.device = "cuda"

    # Initialize tokenizer
    tokenizer = CLIPTokenizer(args.vocab_file, merges_file=args.merges_file)

    # Load models
    models = model_loader.preload_models_from_standard_weights(args.model_file, args.device)

    # Load optional input image
    input_image = None
    if args.input_image_path:
        input_image = Image.open(args.input_image_path)

    # Generate the image using the instance
    output_image = generate(
        prompt=args.prompt,
        uncond_prompt=args.uncond_prompt,
        input_image=input_image,
        strength=args.strength,
        do_cfg=args.do_cfg,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler_name,
        n_inference_steps=args.num_inference_steps,
        seed=args.seed,
        models=models,
        device=args.device,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Display the generated image
    img = Image.fromarray(output_image)
    img.show()
    sleep(120)
    
if __name__ == '__main__':
    main()
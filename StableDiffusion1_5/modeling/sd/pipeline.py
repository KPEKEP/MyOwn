from typing import Dict, Optional, Union
import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from modeling.sd.ddpm import DDPMSampler

# Default values as optional initialization parameters
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_LATENTS_WIDTH = DEFAULT_WIDTH // 8
DEFAULT_LATENTS_HEIGHT = DEFAULT_HEIGHT // 8
DEFAULT_STRENGTH = 0.8
DEFAULT_CFG_SCALE = 7.5
DEFAULT_N_INFERENCE_STEPS = 50
DEFAULT_SEQ_LEN = 77


def generate(
        prompt: str,
        uncond_prompt: Optional[str] = None,
        input_image: Optional[np.ndarray] = None,
        strength: float = DEFAULT_STRENGTH,
        do_cfg: bool = True,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        sampler_name: str = "ddpm",
        n_inference_steps: int = DEFAULT_N_INFERENCE_STEPS,
        models: Dict[str, torch.nn.Module] = {},
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        idle_device: Optional[torch.device] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        latents_height: int = DEFAULT_LATENTS_HEIGHT,
        latents_width: int = DEFAULT_LATENTS_WIDTH,
        seq_len: int = DEFAULT_SEQ_LEN
) -> np.ndarray:
    """
    Generates an image based on a given text prompt and optional input image.

    Parameters:
    - prompt (str): The text prompt to guide the image generation.
    - uncond_prompt (str, optional): Unconditional prompt for cfg.
    - input_image (np.ndarray, optional): An optional input image.
    - strength (float, optional): Strength of the generation, between 0 and 1.
    - do_cfg (bool, optional): Whether to use conditional vs unconditional prompts.
    - cfg_scale (float, optional): Scale factor for cfg.
    - sampler_name (str, optional): Name of the sampler to use.
    - n_inference_steps (int, optional): Number of inference steps.
    - models (dict, optional): Dictionary of pretrained models.
    - seed (int, optional): Random seed.
    - device (torch.device, optional): Device to run the model.
    - idle_device (torch.device, optional): Device to idle the model.
    - tokenizer (PreTrainedTokenizer, optional): Tokenizer for text prompts.
    - latents_height (int, optional): Height of the latent space.
    - latents_width (int, optional): Width of the latent space.
    - seq_len (int, optional): Sequence length for tokenization.

    Returns:
    - np.ndarray: The generated image.
    """
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # Prepare CLIP model
        clip = models.get("clip")
        if clip is None:
            raise ValueError("CLIP model not found in models dictionary.")
        clip.to(device)

        # Tokenization and Context Embedding
        if do_cfg:
            # Conditional and Unconditional Tokenization
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=seq_len
            ).input_ids
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=seq_len
            ).input_ids

            # Shape: (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # Shape: (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)

            # Shape: (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=seq_len
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        # Initialize Sampler
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler value {sampler_name}.")

        # Initialize Latents
        latents_shape = (1, 4, latents_height, latents_width)

        # Conditional path for using input_image
        if input_image:
            encoder = models.get("encoder")
            if encoder is None:
                raise ValueError("Encoder model not found in models dictionary.")
            encoder.to(device)

            input_image_tensor = input_image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Run Diffusion Process
        diffusion = models.get("diffusion")
        if diffusion is None:
            raise ValueError("Diffusion model not found in models dictionary.")
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        # Decode Latents to Images
        decoder = models.get("decoder")
        if decoder is None:
            raise ValueError("Decoder model not found in models dictionary.")
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        # Rescale and Reformat Images
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return images[0]


def rescale(x: torch.Tensor, old_range: tuple, new_range: tuple, clamp: bool = False) -> torch.Tensor:
    """
    Rescales a tensor from one range to another.

    Parameters:
    - x (torch.Tensor): The tensor to rescale.
    - old_range (tuple): The current range of the tensor.
    - new_range (tuple): The new range for the tensor.
    - clamp (bool, optional): Whether to clamp the values to the new range.

    Returns:
    - torch.Tensor: The rescaled tensor.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: float) -> torch.Tensor:
    """
    Generates a time embedding for a given timestep.

    Parameters:
    - timestep (float): The timestep for which to generate the embedding.

    Returns:
    - torch.Tensor: The time embedding tensor.
    """
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
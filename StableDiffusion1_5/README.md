# Inference code for Stable Diffusion v1.5 from scratch

## Command-Line Arguments:

```
python main.py [OPTIONS]

- `--device`: Device to run the models on. Default is `cpu`.
- `--vocab_file`: Path to the vocabulary file for the tokenizer. Default is `data/vocab.json`.
- `--merges_file`: Path to the merges file for the tokenizer. Default is `data/merges.txt`.
- `--model_file`: Path to the model checkpoint file. Default is `data/v1-5-pruned-emaonly.ckpt`.
- `--prompt`: Text prompt for conditional image generation. Default is a predefined string.
- `--uncond_prompt`: Text prompt for unconditional image generation. Default is a predefined string.
- `--do_cfg`: Whether to perform conditional image generation. Default is `True`.
- `--cfg_scale`: Scale factor for conditional image generation. Default is `7.5`.
- `--input_image_path`: Path to an optional input image for conditional image generation.
- `--strength`: Strength of the noise added to the latents. Default is `0.9`.
- `--sampler_name`: Name of the sampler to use. Default is `ddpm`.
- `--num_inference_steps`: Number of inference steps for the sampler. Default is `50`.
- `--seed`: Random seed for reproducibility. Default is `42`.
```

## Prerequisites

To run the code in this repository, you will need to run `download_weights.sh` or download the following files and save them in the `data` directory:

1. **Model Checkpoint File**:  
   - Download `v1-5-pruned-emaonly.ckpt` from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt).

2. **Tokenizer Files**:  
   - Download `merges.txt` from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt).
   - Download `vocab.json` from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/vocab.json).

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments
Thanks to Umar Jamil for a great tutorial video.
Thanks to Hugging Face for providing the model and tokenizer files.
Special thanks to the following repositories:
1. https://github.com/hkproj
2. https://github.com/CompVis/stable-diffusion/
3. https://github.com/divamgupta/stable-diffusion-tensorflow
4. https://github.com/kjsman/stable-diffusion-pytorch
5. https://github.com/huggingface/diffusers/

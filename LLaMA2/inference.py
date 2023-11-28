# Naive LLama2 implementation for inference
# Pavel Nakaznenko, 2023

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class MyOwnLLaMA2:
    """
    A class to encapsulate the transformer model and its tokenizer, 
    and provide text generation functionalities.
    """

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        """
        Initializes the MyOwnLLaMA2 instance with a transformer model, tokenizer, and model arguments.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str,
              tokenizer_path: str,
              load_model: bool,
              max_seq_len: int,
              max_batch_size: int,
              device: str) -> 'MyOwnLLaMA2':
        """
        A factory method to construct a MyOwnLLaMA2 instance.
        """
        # Record the start time for loading the model.
        prev_time = time.time()

        # Load the model checkpoint if specified.
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert checkpoints, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        # Load the model parameters.
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)

        # Construct the model arguments.
        model_args = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device, **params)

        # Load the tokenizer.
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Set the default tensor type based on the device.
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # Construct and load the transformer model.
        model = Transformer(model_args).to(device)
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        return MyOwnLLaMA2(model, tokenizer, model_args)

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Samples a token from the given probabilities using nucleus sampling.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def text_completion(self, 
                        prompts: List[str], 
                        temperature: float = 0.6, 
                        top_p: float = 0.9,
                        max_gen_len: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
        """
        Completes the given text prompts using the transformer model.
        """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Tokenize the prompts.
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len
        total_length = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id

        # Generate tokens to complete the text.
        for cur_pos in tqdm(range(1, total_length), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)  # Greedy decoding.

            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        # Decode the generated tokens to text.
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text


if __name__ == "__main__":
    torch.manual_seed(0)
    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "The Sun is ",
    ]

    model = MyOwnLLaMA2.build(
        checkpoints_dir="llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print("Loaded OK")

    # Inference the model
    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print("-" * 50)
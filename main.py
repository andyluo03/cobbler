from dataclasses import dataclass
from typing import Optional
from safetensors import safe_open
import torch
import json


@dataclass
class Qwen3Config:
  attention_bias: bool
  attention_dropout: float
  bos_token_id: int
  eos_token_id: int
  head_dim: int
  hidden_act: str
  hidden_size: int
  initializer_range: float
  intermediate_size: int
  max_position_embeddings: int
  max_window_layers: int
  num_attention_heads: int
  num_hidden_layers: int
  num_key_value_heads: int
  rms_norm_eps: float
  rope_scaling: Optional[dict]
  rope_theta: int
  sliding_window: Optional[int]
  tie_word_embeddings: bool
  torch_dtype: str
  transformers_version: str
  use_cache: bool
  use_sliding_window: bool
  vocab_size: int

# TODO (andyluo03): support multiple Qwen3 models.
class Qwen3(torch.nn.Module):
    def __init__(self, folder):
        with safe_open(f'{folder}/model-00001-of-00003.safetensors', framework="pt", device="cuda:0") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                print(f"{key}: {shape}")
        
        with safe_open(f'{folder}/model-00002-of-00003.safetensors', framework="pt") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                print(f"{key}: {shape}")

        with safe_open(f'{folder}/model-00003-of-00003.safetensors', framework="pt") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                print(f"{key}: {shape}")

        pass

    def forward(input, batch_size=1) -> list[str]:
        pass

if __name__ == '__main__':
    # Load Model
    model = Qwen3('Qwen3-4B')

    # Run Forwards
    output = model.forward()
    print(output[0])
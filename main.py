from dataclasses import dataclass
from dacite import from_dict
from typing import Optional
from safetensors import safe_open
import torch
import json
from transformers import AutoTokenizer

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


@dataclass
class Qwen3Layer:
    input_layernorm: torch.Tensor
    mlp_down_proj: torch.Tensor
    mlp_gate_proj: torch.Tensor
    mlp_up_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    self_attn_k_norm: torch.Tensor
    self_attn_k_proj: torch.Tensor
    self_attn_o_proj: torch.Tensor
    self_attn_q_norm: torch.Tensor
    self_attn_q_proj: torch.Tensor
    self_attn_v_proj: torch.Tensor
    
    # TODO (andyluo03): implement.
    def forwards(self, inp: torch.Tensor, out: torch.Tensor):
        # [Sequence, Embedding Size] --> [Sequence, Embedding Size]
        pass

# TODO (andyluo03): support multiple Qwen3 models.
class Qwen3:
    def __init__(self, folder):
        # 0. Load Metadata
        with open(f'{folder}/config.json') as f:
            config = json.load(f)
            self.config = from_dict(data_class=Qwen3Config, data=config)

        with open(f'{folder}/model.safetensors.index.json') as f:
            tensor_index = json.load(f)

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True)
        
        # 2. Load Embedding And Norm Weights
        def load_weight(name):
            with safe_open(f'{folder}/{tensor_index['weight_map'][name]}', framework="pt", device="cuda:0") as f:
                return f.get_tensor(name) # tensor persists, slice lazy loads

        self.embed_tokens = load_weight('model.embed_tokens.weight')
        self.norm = load_weight('model.norm.weight')
        
        # 3. Load Hidden Layer Weights
        self.hidden_layers = []

        hidden_weights = [
            "input_layernorm",
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "post_attention_layernorm",
            "self_attn.k_norm",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.q_proj",
            "self_attn.v_proj"
        ]

        for i in range(self.config.num_hidden_layers):
            prefix = f'model.layers.{i}.'

            layer_data = {
                "input_layernorm":          load_weight(f'{prefix}input_layernorm.weight'),
                "mlp_down_proj":            load_weight(f'{prefix}mlp.down_proj.weight'),
                "mlp_gate_proj":            load_weight(f'{prefix}mlp.gate_proj.weight'),
                "mlp_up_proj":              load_weight(f'{prefix}mlp.up_proj.weight'),
                "post_attention_layernorm": load_weight(f'{prefix}post_attention_layernorm.weight'),
                "self_attn_k_norm":         load_weight(f'{prefix}self_attn.k_norm.weight'),
                "self_attn_k_proj":         load_weight(f'{prefix}self_attn.k_proj.weight'),
                "self_attn_o_proj":         load_weight(f'{prefix}self_attn.o_proj.weight'),
                "self_attn_q_norm":         load_weight(f'{prefix}self_attn.q_norm.weight'),
                "self_attn_q_proj":         load_weight(f'{prefix}self_attn.q_proj.weight'),
                "self_attn_v_proj":         load_weight(f'{prefix}self_attn.v_proj.weight'),
            }

            self.hidden_layers.append(Qwen3Layer(**layer_data))

    def forward(self, input: list[int], batch_size=1) -> list[int]:
        input = torch.Tensor()
        output = torch.Tensor()

        # 0. Embed

        # 1. Run Hidden Layers
        for hidden_layer in range(self.hidden_layers):
            hidden_layer.forward(input, output)
            input, output = output, input

        # 2. Generate Distribution

    def print_summary(self):
        pass

if __name__ == '__main__':
    # 0. Load Model
    model = Qwen3('Qwen3-4B')

    # 1. Run Model
    prompt = 'The meaning of life is'
    tokens = model.tokenizer(prompt)['input_ids'] 

    while True: 
        # output_distribution = model.forward(input=tokens)
        # TODO (andyluo03): sample from distribution + append to tokens
        pass

    print(final_output)
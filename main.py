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


def apply_rotary_emb(x: torch.Tensor, theta: float) -> torch.Tensor:
    # x shape: [seq_len, num_heads, head_dim]
    seq_len, num_heads, head_dim = x.shape
    device = x.device
    
    # 1. Generate frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq) # [seq_len, head_dim // 2]
    
    # 2. Create cos/sin and unsqueeze for heads
    # Resulting shape: [seq_len, 1, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(seq_len, 1, head_dim) 
    sin = emb.sin().view(seq_len, 1, head_dim)
    
    # 3. Rotate x
    # x1/x2 are [seq_len, num_heads, head_dim // 2]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    
    # Broadcasting: [seq_len, num_heads, head_dim] * [seq_len, 1, head_dim]
    return (x * cos) + (rotated * sin)


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
    
    def forwards(self, inp: torch.Tensor, out: torch.Tensor, config: Qwen3Config):
            # 1. RMSNorm + Self Attention
            residual = inp
            hidden_states = rmsnorm(inp, self.input_layernorm, config.rms_norm_eps)
            
            # --- Multi-Head Attention ---
            seq_len, _ = hidden_states.shape
            
            # Projections
            q = torch.matmul(hidden_states, self.self_attn_q_proj.t())
            k = torch.matmul(hidden_states, self.self_attn_k_proj.t())
            v = torch.matmul(hidden_states, self.self_attn_v_proj.t())

            # Reshape for multi-head (GQA support)
            # Q: [Seq, Num_Heads, Head_Dim] | K/V: [Seq, Num_KV_Heads, Head_Dim]
            q = q.view(seq_len, config.num_attention_heads, config.head_dim)
            k = k.view(seq_len, config.num_key_value_heads, config.head_dim)
            v = v.view(seq_len, config.num_key_value_heads, config.head_dim)

            # Apply QK Norm (Typical in Qwen architectures)
            q = rmsnorm(q, self.self_attn_q_norm, config.rms_norm_eps)
            k = rmsnorm(k, self.self_attn_k_norm, config.rms_norm_eps)

            # Apply RoPE
            q = apply_rotary_emb(q, config.rope_theta)
            k = apply_rotary_emb(k, config.rope_theta)

            # Grouped Query Attention (repeat K/V heads to match Q heads)
            num_groups = config.num_attention_heads // config.num_key_value_heads
            if num_groups > 1:
                k = k.repeat_interleave(num_groups, dim=1)
                v = v.repeat_interleave(num_groups, dim=1)

            # Scaled Dot-Product Attention
            # Transpose to [Heads, Seq, Head_Dim] for matmul
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            
            # [Heads, Seq, Seq]
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (config.head_dim ** 0.5)
            
            # Causal Mask
            mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device).triu(1)
            attn_weights += mask
            
            attn_weights = torch.softmax(attn_weights, dim=-1).to(hidden_states.dtype)
            
            # [Heads, Seq, Head_Dim] -> [Seq, Hidden_Size]
            attn_output = torch.matmul(attn_weights, v).transpose(0, 1).contiguous()
            attn_output = attn_output.view(seq_len, -1)
            
            # Output Projection
            attn_output = torch.matmul(attn_output, self.self_attn_o_proj.t())
            
            hidden_states = residual + attn_output

            # 2. RMSNorm + FFN (MLP)
            residual = hidden_states
            hidden_states = rmsnorm(hidden_states, self.post_attention_layernorm, config.rms_norm_eps)
            
            # 3. SwiGLU: (SiLU(Gate) * Up) * Down
            gate = torch.matmul(hidden_states, self.mlp_gate_proj.t())
            up = torch.matmul(hidden_states, self.mlp_up_proj.t())
            intermediate = torch.nn.functional.silu(gate) * up
            ffn_output = torch.matmul(intermediate, self.mlp_down_proj.t())

            # 4. Final Residual
            out.copy_(residual + ffn_output)


def rmsnorm(x, weight, eps=1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


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
            weight_file = tensor_index["weight_map"][name]
            with safe_open(f"{folder}/{weight_file}", framework="pt", device="cuda:0") as f:
                return f.get_tensor(name) # tensor persists, slice lazy loads

        self.embed_tokens = load_weight('model.embed_tokens.weight')
        self.norm = load_weight('model.norm.weight')
        
        # 3. Load Hidden Layer Weights
        self.hidden_layers = []

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

    def forwards(self, input: list[int], batch_size=1) -> torch.Tensor:
        # 0. Get Sequence Embedding
        # -- From my understanding, this is essentially a lookup table.
        dinput = torch.tensor(input, dtype=torch.long, device='cuda:0')

        # [Seqlen, Hidden Dimension]
        linput  = torch.nn.functional.embedding(dinput, self.embed_tokens)
        loutput = torch.zeros_like(linput)

        # 1. Run Hidden Layers
        for hidden_layer in self.hidden_layers:
            hidden_layer.forwards(linput, loutput, self.config)
            linput, loutput = loutput, linput

        # 2. Generate Distribution

        # 2a. RMSNorm TODO (andyluo03): 'scratch-ify' this.
        normed = rmsnorm(linput[-1:, :], self.norm, self.config.rms_norm_eps)

        # 2b. Linear Layer 
        # -- Qwen3 uses "weight tying" --> Linear Layer == Embed Tokens
        logits = torch.matmul(normed, self.embed_tokens.t())

        return logits.squeeze(0)

    def print_summary(self):
        pass

if __name__ == '__main__':
    # 0. Load Model
    model = Qwen3('Qwen3-4B')
    print('Model weights loaded!')

    # 1. Run Model
    prompt = 'The meaning of life is'
    tokens = model.tokenizer(prompt)['input_ids'] 

    max_tokens = 50
    for _ in range(max_tokens): 
        # 1a. Run Forwards
        logits = model.forwards(tokens)
        
        # 1b. Sample Logits
        latest_token_tensor = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        latest_token = latest_token_tensor.item()

        # 1c. Handle Token
        if latest_token == model.config.eos_token_id:
            break

        tokens.append(latest_token)

    # 2. Detokenize Model
    final_output = model.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f'Model Output:\n{final_output}')
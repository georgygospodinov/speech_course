import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(
            self,
            n_q: int,
            n_embd: int,
            n_layer: int,
            n_head: int,
            n_query_groups: int,
            head_size: int,
            intermediate_size: int,
            add_qkv_bias: int,
            bias: bool,
            block_size: int,
            norm_eps: float,
            tie_word_embeddings: bool,
            scale_embeddings: bool,
            text_vocab_size: int,
            audio_vocab_size: int,
            lm_head_bias: bool,
            rotary_percentage: float,
            rope_condense_ratio: int,
            rope_base: int,
            **kwargs
            ) -> None:
        super().__init__()
        self.lm_heads = nn.ModuleList([nn.Linear(n_embd, text_vocab_size, bias=lm_head_bias)] + [nn.Linear(n_embd, audio_vocab_size, bias=lm_head_bias) for _ in range(n_q)])
        self.embeds = nn.ModuleList([nn.Embedding(text_vocab_size, n_embd)] + [nn.Embedding(audio_vocab_size, n_embd) for _ in range(n_q)])
        self.ln = RMSNorm(n_embd, eps=norm_eps)
        self.rope_n_elem = int(rotary_percentage * head_size)
        self.transformer = nn.ModuleList(Block(
            n_embd, 
            norm_eps, 
            n_head, 
            n_query_groups, 
            head_size, 
            intermediate_size, 
            add_qkv_bias, 
            self.rope_n_elem, 
            bias
            ) for _ in range(n_layer))
        
        self.block_size = block_size
        self.scale_embeddings = scale_embeddings
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_base = rope_base
        self.n_embd = n_embd
        self.n_q = n_q
        self.max_seq_length = block_size
        self.mask_cache: Optional[torch.Tensor] = None
        if tie_word_embeddings:
            for lm_head, emb in zip(self.lm_heads, self.embeds):
               lm_head.weight = emb.weight

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.block_size}"
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        bs, _, T = input_ids.shape
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."
            )
            
        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos.view(-1)).view(*input_pos.shape, -1)
            sin = self.sin.index_select(0, input_pos.view(-1)).view(*input_pos.shape, -1)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(0, input_pos.view(-1)).view(bs, 1, input_pos.shape[1], -1)
        else:
            cos = self.cos[None, :T, :].repeat(bs, 1, 1)
            sin = self.sin[None, :T, :].repeat(bs, 1, 1)
            mask = None

        x = 0
        for i, emb in enumerate(self.embeds):
            x += emb(input_ids[:, i])
        x = x / (self.n_q + 1)

        if self.scale_embeddings:
            x = x * (self.n_embd**0.5)

        for block in self.transformer:
            x = block(x, cos, sin, mask, input_pos)

        x_ori = x
        x_ori = self.ln(x_ori)
        
        logits = []
        for lm_head in self.lm_heads:
            logits.append(lm_head(x_ori))

        return logits

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.rope_n_elem,
            device=device,
            condense_ratio=self.rope_condense_ratio,
            base=self.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, n_embd: int, norm_eps: float, n_head: int, n_query_groups: int, head_size: int, intermediate_size: int, add_qkv_bias: bool, rope_n_elem: int, bias: bool) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(n_embd, eps=norm_eps)
        self.attn = CausalSelfAttention(n_embd, n_head, n_query_groups, head_size, add_qkv_bias, rope_n_elem, bias)
        self.norm_2 = RMSNorm(n_embd, eps=norm_eps)
        self.mlp = LLaMAMLP(n_embd, intermediate_size, bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        x = attention_output + x
        x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_query_groups: int, head_size: int, add_qkv_bias: bool, rope_n_elem: int, bias: bool=False) -> None:
        super().__init__()
        shape = (n_head + 2 * n_query_groups) * head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=add_qkv_bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            head_size * n_head, n_embd, bias=bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_size = head_size
        self.rope_n_elem = rope_n_elem

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.n_query_groups, total_qkv, self.head_size
        )
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.n_query_groups != self.n_head and (
            input_pos is None or self.n_query_groups != 1
        ):
            k = k.expand(
                B, self.n_query_groups, q_per_kv, T, self.head_size
            )
            v = v.expand(
                B, self.n_query_groups, q_per_kv, T, self.head_size
            )

        q = q.reshape(B, -1, T, self.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(
            B, T, self.head_size * self.n_head
        )  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.n_query_groups == 1 else self.n_head
        v_shape = (batch_size, heads, max_seq_length, self.head_size)
        if rope_cache_length is None:
            if self.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.head_size - self.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        ks = []
        vs = []
        for i in range(input_pos.shape[0]):
            ks.append(self.k[i].index_copy_(1, input_pos[i], k[i]))
            vs.append(self.v[i].index_copy_(1, input_pos[i], v[i]))
        return torch.stack(ks, 0), torch.stack(vs, 0)

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(
    max_seq_length: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos[:, None]) + (rotated * sin[:, None])
    return roped.to(dtype=x.dtype)


class LLaMAMLP(nn.Module):
    def __init__(self, n_embd: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.fc_2 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.proj = nn.Linear(intermediate_size, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)
    

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        if self.add_unit_offset:
            # Gemma model requires a unit offset
            # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
            return x_normed * (1 + self.weight)
        return x_normed * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

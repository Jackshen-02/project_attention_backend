import numpy as np
import torch
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False, attention_backend: Optional[str]=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        # COPY FROM ASSIGN2_4
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        self.use_fused_kernel = use_fused_kernel
        if attention_backend is None:
            self.attention_backend = "hw4_fused" if use_fused_kernel else "naive"
        else:
            self.attention_backend = attention_backend

    def _torch_device(self, preferred_device: Optional[torch.device] = None) -> torch.device:
        if preferred_device is not None:
            return preferred_device
        if self.backend is not None and getattr(self.backend, "cuda", False) and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _project_to_torch_qkv(self, x, preferred_device: Optional[torch.device] = None):
        q, kT, v = self.project_to_query_key_value(x)
        return self._qkv_to_torch(q, kT, v, preferred_device=preferred_device)

    def _qkv_to_torch(self, q, kT, v, preferred_device: Optional[torch.device] = None):
        device = self._torch_device(preferred_device)
        q_t = torch.tensor(q.to_numpy(), dtype=torch.float32, device=device)
        k_t = torch.tensor(
            kT.permute(0, 1, 3, 2).contiguous().to_numpy(),
            dtype=torch.float32,
            device=device,
        )
        v_t = torch.tensor(v.to_numpy(), dtype=torch.float32, device=device)
        return q_t, k_t, v_t

    def create_causal_mask(self, bs, nh, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        # mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1) # This should be ok, but may be problematic -> the loss will be NaN in Assignment 3 because the mask will not broadcast correctly in the kernel.
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        x = x.contiguous()
        
        # COPY FROM ASSIGN2_4
        x_2d = x.view(batch_size * seq_len, n_embd)
        q = self.q_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        k = self.k_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        v = self.v_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        kT = k.permute(0, 1, 3, 2)
        
        return q, kT, v

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        if self.attention_backend == "flash_tiled":
            return self._flash_tiled_attention(q, kT, v)

        scores = (q @ kT) / np.sqrt(q_dim)

        if self.attention_backend == "naive":
            if self.causal:
                scores = scores + self.create_causal_mask(batch_size, num_head, queries_len)
            attn = softmax(scores, dim=3)
        elif self.attention_backend == "hw4_fused":
            if self.causal:
                attn = scores.attn_softmax()
            else:
                attn = softmax(scores, dim=3)
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        attn = self.dropout(attn)
        out = attn @ v
        result = out.permute(0, 2, 1, 3).contiguous().view(batch_size, queries_len, num_head * q_dim)
        return result

    def _flash_tiled_attention(self, q, kT, v):
        from attention_backend.flash import FlashAttentionConfig, flash_attention_tiled

        batch_size, num_head, queries_len, q_dim = q.shape
        q_t, k_t, v_t = self._qkv_to_torch(q, kT, v)

        out_t = flash_attention_tiled(
            q_t,
            k_t,
            v_t,
            config=FlashAttentionConfig(causal=self.causal),
        )
        out_np = out_t.detach().cpu().numpy().astype(datatype)
        out = tensor_from_numpy(out_np, backend=self.backend)
        return out.permute(0, 2, 1, 3).contiguous().view(batch_size, queries_len, num_head * q_dim)

    def decode_step(self, x, kv_cache: Any, *, cache_backend: str = "contiguous"):
        from attention_backend.paged import decode_attention_contiguous, decode_attention_paged

        batch_size, seq_len, n_embd = x.shape
        if seq_len != 1:
            raise ValueError("decode_step expects x with shape (batch_size, 1, n_embd).")

        cache_device = getattr(getattr(kv_cache, "keys", None), "device", None)
        q_t, k_t, v_t = self._project_to_torch_qkv(x, preferred_device=cache_device)
        kv_cache.append(k_t, v_t)

        if cache_backend == "contiguous":
            out_t = decode_attention_contiguous(q_t, kv_cache)
        elif cache_backend == "paged":
            out_t = decode_attention_paged(q_t, kv_cache)
        else:
            raise ValueError(f"Unknown decode cache backend: {cache_backend}")

        out_np = out_t.detach().cpu().numpy().astype(datatype)
        out = tensor_from_numpy(out_np, backend=self.backend)
        result = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.n_embd)
        result = self.out_projection(result.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        return result

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed. 

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # COPY FROM ASSIGN2_4
        q, kT, v = self.project_to_query_key_value(x)
        result = self.self_attention(q, kT, v)
        result = self.out_projection(result.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        return result


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        # COPY FROM ASSIGN2_4
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM ASSIGN2_4
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False, attention_backend: Optional[str]=None):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        
        # COPY FROM ASSIGN2_4
        # self.attention
        # self.ff
        self.attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel, attention_backend=attention_backend)
        self.ff = FeedForward(n_embd=n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.use_fused_kernel = use_fused_kernel

        # if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln_1
            # self.ln_2
            
        # else:
            # BEGIN ASSIGN3_3
            # END ASSIGN3_3

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            x_ln = self.ln_1(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim)
            x = x + self.attention(x_ln)
            x_ln = self.ln_2(x.view(batch_size * seq_len, x_dim)).view(batch_size, seq_len, x_dim)
            x = x + self.ff(x_ln)
        else:
            # BEGIN ASSIGN3_3
            x_ln = x.view(batch_size * seq_len, x_dim).layernorm(
                self.ln_1.weights.value,
                self.ln_1.bias.value,
            ).view(batch_size, seq_len, x_dim)
            x = x + self.attention(x_ln)

            x_ln = x.view(batch_size * seq_len, x_dim).layernorm(
                self.ln_2.weights.value,
                self.ln_2.bias.value,
            ).view(batch_size, seq_len, x_dim)
            x = x + self.ff(x_ln)
            # END ASSIGN3_3

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
        attention_backend: Optional[str]=None,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.
        
        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        
        # COPY FROM ASSIGN2_4
        self.token_embeddings = Embedding(num_embeddings=n_vocab, embedding_dim=n_embd, backend=backend)
        self.position_embeddings = Embedding(num_embeddings=n_positions, embedding_dim=n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel, attention_backend=attention_backend)
        self.t_layer_2 = TransformerLayer(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel, attention_backend=attention_backend)
        self.t_layer_3 = TransformerLayer(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel, attention_backend=attention_backend)
        self.t_layer_4 = TransformerLayer(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_fused_kernel=use_fused_kernel, attention_backend=attention_backend)
        self.dropout = Dropout(p_dropout=p_dropout)
        self.ln = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=backend)
        self.lm_head = Linear(in_size=n_embd, out_size=n_vocab, bias=bias, backend=backend)

        self.use_fused_kernel = use_fused_kernel
        # if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
        # else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            # END ASSIGN3_3
        
    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        token_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(pos)
        x = token_emb + pos_emb
        x = self.dropout(x)
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)

        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_embd)
            
        else:
            # BEGIN ASSIGN3_3
            x = x.view(batch_size * seq_len, self.n_embd).layernorm(self.ln.weights.value, self.ln.bias.value).view(batch_size, seq_len, self.n_embd)
            # END ASSIGN3_3

        logits = self.lm_head(x.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_vocab)
        # return x
        return logits

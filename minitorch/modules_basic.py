"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        
        # COPY FROM ASSIGN2_3
        self.weights = Parameter(tensor_from_numpy(np.random.randn(num_embeddings, embedding_dim), backend=backend))
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        
        # COPY FROM ASSIGN2_3
        x_one_hot = one_hot(x, self.num_embeddings)                 # (bs, seq_len, num_embeddings)
        x_2d = x_one_hot.view(bs * seq_len, self.num_embeddings)    # (bs * seq_len, num_embeddings)
        out_2d = x_2d @ self.weights.value      # (bs * seq_len, num_embeddings) @ (num_embeddings, embedding_dim) -> (bs * seq_len, embedding_dim)
        return out_2d.view(bs, seq_len, self.embedding_dim)         # (bs, seq_len, embedding_dim)

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        # COPY FROM ASSIGN2_3
        if (not self.training) or self.p_dropout == 0.0:
            return x
        
        if self.p_dropout >= 1.0:
            return zeros(x.shape, backend=x.backend)
        
        keep_prob = 1.0 - self.p_dropout
        mask = tensor_from_numpy(np.random.binomial(1, keep_prob, size=x.shape), backend=x.backend)
        return x * mask / keep_prob


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        
        # COPY FROM ASSIGN2_3
        self.backend = backend
        limit = 1.0 / np.sqrt(in_size)
        self.weights = Parameter((2.0 * rand((in_size, out_size), backend=backend) - 1.0) * limit)
        if bias:
            self.bias = Parameter((2.0 * rand((out_size,), backend=backend) - 1.0) * limit)
        else:
            self.bias = None

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        
        # COPY FROM ASSIGN2_3
        x_ = x.view(batch, in_size)
        w_ = self.weights.value.view(in_size, self.out_size)
        out = (x_ @ w_).view(batch, self.out_size)
        if self.bias is not None:
            out = out + self.bias.value
        return out


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        
        # COPY FROM ASSIGN2_3
        self.backend = backend
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        
        # COPY FROM ASSIGN2_3
        mean = x.mean(dim=1)                                # (bs, 1)
        var = ((x - mean) * (x - mean)).mean(dim=1)         # (bs, 1)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)      # (bs, dim)
        return x_hat * self.weights.value + self.bias.value

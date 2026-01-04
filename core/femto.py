"""
Konomi Corduroy-OS - Femto Module
16-dimensional nano LLM with JAX backend

🧠 Femto: Ultra-lightweight language model for embedded AI nodes
"""

import jax
import jax.numpy as jnp
from jax import random
import hashlib


class Femto:
    """
    16-dimensional nano language model.

    A minimal LLM designed for deployment at every coordinate
    in the BlockArray 3D grid and at each vertex of the Cube.

    Architecture:
    - 16x16 weight matrix (256 parameters)
    - Hash-based text encoding
    - Simple forward pass
    - <1MB memory per instance

    Targets:
    - 0.1s per request
    - 4MB total footprint including overhead
    - Can run thousands of instances concurrently
    """

    def __init__(self, dim=16, seed=0):
        """
        Initialize Femto LLM.

        Args:
            dim: Embedding dimension (default 16)
            seed: Random seed for weight initialization
        """
        self.dim = dim
        self.key = random.PRNGKey(seed)

        # Initialize weight matrix: 16x16
        self.W = random.normal(self.key, (dim, dim)) * 0.1

        # Bias vector
        self.b = jnp.zeros(dim)

        # Instance ID for debugging
        self.id = hashlib.md5(str(seed).encode()).hexdigest()[:8]

    def p(self, text):
        """
        Process text (primary interface).

        For MVP: Returns formatted text with Femto marker.
        Future: Will encode, process through network, decode.

        Args:
            text: Input text string

        Returns:
            Processed text string

        Examples:
            >>> femto = Femto()
            >>> femto.p("Hello world")
            '[Hello world]'
        """
        # Truncate to 50 chars and wrap in brackets
        return f"[{text[:50]}]"

    def forward(self, x):
        """
        Forward pass through the nano network.

        Args:
            x: Input vector (16-dimensional)

        Returns:
            Output vector after transformation

        Math: y = (x @ W) + b
        """
        x_jax = jnp.asarray(x)
        return jnp.dot(x_jax, self.W) + self.b

    def encode_text(self, text):
        """
        Encode text to 16-dimensional vector.

        Uses hash-based encoding for simplicity. Each character
        contributes to the final vector through hashing.

        Args:
            text: Input text string

        Returns:
            16-dimensional JAX array

        Examples:
            >>> femto = Femto()
            >>> vec = femto.encode_text("test")
            >>> vec.shape
            (16,)
        """
        # Hash text to get deterministic integer
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # Create vector with sparse activation
        vec = jnp.zeros(self.dim)

        # Set multiple positions based on hash
        for i in range(4):  # Use 4 positions for richer encoding
            pos = (h >> (i * 4)) % self.dim
            val = ((h >> (i * 8)) % 100) / 100.0  # Value between 0-1
            vec = vec.at[pos].add(val)

        # Normalize
        norm = jnp.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def process_full(self, text):
        """
        Full processing pipeline: encode -> forward -> decode.

        Args:
            text: Input text string

        Returns:
            Processed embedding vector

        Examples:
            >>> femto = Femto()
            >>> result = femto.process_full("analyze this")
            >>> result.shape
            (16,)
        """
        # Encode text to vector
        x = self.encode_text(text)

        # Forward pass
        y = self.forward(x)

        return y

    def similarity(self, text1, text2):
        """
        Compute similarity between two texts.

        Uses cosine similarity of processed embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        v1 = self.process_full(text1)
        v2 = self.process_full(text2)

        # Cosine similarity
        dot = jnp.dot(v1, v2)
        norm1 = jnp.linalg.norm(v1)
        norm2 = jnp.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def update_weights(self, x, y, lr=0.01):
        """
        Simple weight update (for future learning).

        Args:
            x: Input vector
            y: Target output vector
            lr: Learning rate

        Returns:
            Loss value
        """
        # Forward pass
        pred = self.forward(x)

        # MSE loss
        loss = jnp.mean((pred - y) ** 2)

        # Gradient (simplified - full version would use JAX autodiff)
        # For MVP, just track loss
        return float(loss)

    def info(self):
        """
        Get Femto instance information.

        Returns:
            Dict with instance details
        """
        return {
            'id': self.id,
            'dim': self.dim,
            'params': self.W.size + self.b.size,
            'weight_shape': self.W.shape,
            'memory_estimate_kb': (self.W.nbytes + self.b.nbytes) / 1024
        }

    def __repr__(self):
        return f"Femto(id={self.id}, dim={self.dim}, params={self.W.size + self.b.size})"

"""Shared neural network building blocks (JAX/Flax nnx)."""

from flax import nnx


class MLP(nnx.Module):
    """Multi-layer perceptron with input LayerNorm.

    Architecture: LayerNorm -> [Linear -> ReLU] x num_hidden_layers -> Linear

    Used by the actor and critic networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        rngs: nnx.Rngs,
        final_kernel_init=None,
        final_bias_init=None,
    ) -> None:
        self.norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self._num_hidden = num_hidden_layers

        prev_dim = input_dim
        for i in range(num_hidden_layers):
            setattr(self, f"fc{i}", nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        setattr(
            self,
            f"fc{num_hidden_layers}",
            nnx.Linear(
                prev_dim, output_dim,
                kernel_init=final_kernel_init or nnx.initializers.lecun_normal(),
                bias_init=final_bias_init or nnx.initializers.zeros,
                rngs=rngs,
            ),
        )

    def __call__(self, x):
        x = self.norm(x)
        for i in range(self._num_hidden):
            x = getattr(self, f"fc{i}")(x)
            x = nnx.relu(x)
        x = getattr(self, f"fc{self._num_hidden}")(x)
        return x

from .attention import AttentionEncoder
from .base import Encoder
from .linear import LinearEncoder
from .mlp import MLPEncoder

ENCODER_REGISTRY = {
    "linear": LinearEncoder,
    "mlp": MLPEncoder,
    "attention": AttentionEncoder,
}


def build_encoder(encoder_type: str, **kwargs) -> Encoder:
    return ENCODER_REGISTRY[encoder_type](**kwargs)


__all__ = ["Encoder", "LinearEncoder", "MLPEncoder", "AttentionEncoder", "build_encoder"]

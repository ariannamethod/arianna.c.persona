"""
Arianna - Post-Transformer Language Organism

Proof of concept: Transformer architecture WITHOUT pretrained weights
Knowledge lives in dynamic numpy shards, not static parameters

Only dependency: numpy
"""

from .arianna import Arianna
from .transformer import MinimalTransformer, TransformerConfig
from .shard_manager import ShardManager, NumpyShard
from .tokenizer import DynamicTokenizer
from .presence import PresenceComputer, PresencePulse

__version__ = "0.1.0"
__all__ = [
    "Arianna",
    "MinimalTransformer",
    "TransformerConfig",
    "ShardManager",
    "NumpyShard",
    "DynamicTokenizer",
    "PresenceComputer",
    "PresencePulse",
]

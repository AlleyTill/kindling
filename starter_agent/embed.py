"""Embedders turn text into a list of numbers, so similar texts get similar lists.

HashEmbedder is the default: it needs nothing and downloads nothing, but it only sees shared words and
spellings, not meaning. For meaning, install sentence-transformers and set "embedder": "bge-small";
the first use downloads the bge-small-en-v1.5 model (about 130 MB) once.
"""
import math
import re
import zlib
from array import array

WORD = re.compile(r"\w+")


def words(text):
    return WORD.findall(text.lower())


def normalize(vec):
    length = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / length for x in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(normalize(a), normalize(b)))


def to_blob(vec):
    return array("f", vec).tobytes()


def from_blob(blob):
    vec = array("f")
    vec.frombytes(blob)
    return list(vec)


class HashEmbedder:
    """Each word, and each three-letter piece of it, is hashed to one of `dim` slots."""

    def __init__(self, dim=256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts):
        return [self._one(text) for text in texts]

    def _one(self, text):
        vec = [0.0] * self.dim
        for word in words(text):
            padded = f"<{word}>"
            pieces = [word] * 2 + [padded[i:i + 3] for i in range(len(padded) - 2)]
            for piece in pieces:
                h = zlib.crc32(piece.encode())       # the same on every machine, unlike hash()
                vec[h % self.dim] += 1.0 if h & 1 else -1.0
        return normalize(vec)


class SentenceTransformerEmbedder:
    def __init__(self, model="BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit('The bge-small embedder needs `pip install sentence-transformers`; '
                             'or set "embedder": "hash" in agent.json.')
        self.model = SentenceTransformer(model)
        self.name = model

    def embed(self, texts):
        return [[float(x) for x in v] for v in self.model.encode(list(texts), normalize_embeddings=True)]


def make_embedder(name):
    if name == "hash":
        return HashEmbedder()
    if name == "bge-small":
        return SentenceTransformerEmbedder()
    raise SystemExit(f'Unknown embedder "{name}": use "hash" or "bge-small".')

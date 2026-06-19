"""
Tokenizer wrapper for the Transformer project.
Uses EthioBBPE under the hood — scope: text → token IDs only.
"""

from typing import List, Union
from ethiobbpe import EthioBBPETokenizer as _EthioBBPETokenizer


class Tokenizer:
    """
    Thin, import-friendly wrapper around EthioBBPETokenizer.

    Usage
    -----
        from tokenizer import Tokenizer

        tok = Tokenizer()
        ids   = tok.encode("ሰላም ዓለም")          # List[int]
        batch = tok.encode_batch(["ሰላም", "ዓለም"]) # List[List[int]]
    """

    def __init__(self) -> None:
        # Auto-downloads model from Hugging Face on first run; cached afterwards.
        self._tok = _EthioBBPETokenizer.from_pretrained()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        truncation: bool = False,
        max_length: int = 512,
    ) -> List[int]:
        """
        Tokenize a single string and return token IDs.

        Parameters
        ----------
        text        : input string
        truncation  : whether to truncate to max_length
        max_length  : maximum sequence length (used only when truncation=True)

        Returns
        -------
        List[int]   : token IDs
        """
        encoding = self._tok.encode(
            text,
            truncation=truncation,
            max_length=max_length,
        )
        return encoding.ids

    def encode_batch(
        self,
        texts: List[str],
        truncation: bool = False,
        max_length: int = 512,
    ) -> List[List[int]]:
        """
        Tokenize a list of strings and return a list of token ID sequences.

        Parameters
        ----------
        texts       : list of input strings
        truncation  : whether to truncate each sequence to max_length
        max_length  : maximum sequence length (used only when truncation=True)

        Returns
        -------
        List[List[int]] : token IDs for each text
        """
        encodings = self._tok.encode_batch(
            texts,
            truncation=truncation,
            max_length=max_length,
        )
        return [enc.ids for enc in encodings]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[str, List[str]],
        truncation: bool = False,
        max_length: int = 512,
    ) -> Union[List[int], List[List[int]]]:
        """
        Callable shorthand:
            tok("ሰላም")          → List[int]
            tok(["ሰላም", "ዓለም"]) → List[List[int]]
        """
        if isinstance(text, list):
            return self.encode_batch(text, truncation=truncation, max_length=max_length)
        return self.encode(text, truncation=truncation, max_length=max_length)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (16 000)."""
        return self._tok.get_vocab_size()

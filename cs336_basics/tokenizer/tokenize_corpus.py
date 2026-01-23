import os
import base64
import json
import regex as re
from collections import defaultdict
from collections import Counter
from typing import Iterable, Iterator

_WORKER_SPLIT_RE = None
_WORKER_PAT_RE = None
_PAT_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.byte_to_id = {tok_bytes: tok_id for tok_id, tok_bytes in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._special_bytes = {t.encode("utf-8") for t in self.special_tokens}
        if self.special_tokens:
            # Sort longer tokens first so overlapping specials prefer the longest match.
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("|".join(re.escape(t) for t in specials_sorted))
            self._max_special_len = max(len(t) for t in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0
        self._bpe_cache: dict[bytes, list[bytes]] = {}
        
        def default_tok_value():
            return (0, 0)
        
        self.tokens = defaultdict(default_tok_value) # str -> (int, int) : preserves pre-token counts and indexes in a tuple
    # will be implemented later
    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)

    def _bpe_encode_bytes(self, token_bytes: bytes) -> list[bytes]:
        # Start from byte-level tokens, iteratively merge best-ranked pairs.
        if not token_bytes:
            return []
        cached = self._bpe_cache.get(token_bytes)
        if cached is not None:
            return cached
        tokens = [bytes([b]) for b in token_bytes]
        while True:
            best_rank = None
            best_pair = None
            for a, b in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get((a, b))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (a, b)
            if best_pair is None:
                break
            merged = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        self._bpe_cache[token_bytes] = tokens
        return tokens

    def _tokenize_segment(self, segment: str, defer_last: bool) -> tuple[list[bytes], str]:
        tokens = []
        last = None
        for m in _PAT_RE.finditer(segment):
            if last is not None:
                tokens.append(last.group().encode("utf-8"))
            last = m
        if last is None:
            return tokens, ""
        if defer_last and last.end() == len(segment):
            return tokens, last.group()
        tokens.append(last.group().encode("utf-8"))
        return tokens, ""

    def _pretokenize_chunk(self, text_chunk: str, final: bool) -> tuple[list[bytes], str]:
        if self._special_re is None:
            return self._tokenize_segment(text_chunk, defer_last=not final)

        if final:
            process_text = text_chunk
            tail = ""
        else:
            # Keep enough tail to avoid splitting a special token across chunks.
            tail_len = max(self._max_special_len, 0)
            cut = max(0, len(text_chunk) - tail_len)
            process_text = text_chunk[:cut]
            tail = text_chunk[cut:]

        tokens: list[bytes] = []
        last_end = 0
        for sm in self._special_re.finditer(process_text):
            segment = process_text[last_end:sm.start()]
            seg_tokens, _ = self._tokenize_segment(segment, defer_last=False)
            tokens.extend(seg_tokens)
            tokens.append(sm.group().encode("utf-8"))
            last_end = sm.end()
        segment = process_text[last_end:]
        seg_tokens, remainder = self._tokenize_segment(segment, defer_last=not final)
        tokens.extend(seg_tokens)
        return tokens, remainder + tail

    def _encode_stream_chunk(self, text_chunk: str, final: bool) -> tuple[list[int], str]:
        pretoken_bytes, remainder = self._pretokenize_chunk(text_chunk, final=final)
        token_ids: list[int] = []
        for token_bytes in pretoken_bytes:
            if token_bytes in self._special_bytes:
                token_ids.append(self.byte_to_id[token_bytes])
                continue
            for bpe_token in self._bpe_encode_bytes(token_bytes):
                token_ids.append(self.byte_to_id[bpe_token])
        return token_ids, remainder

    def encode(self, text: str) -> list[int]:
        """
        Step 1: pre-tokenize the sequence and represent each pre-token as a sequence of UTF-8 bytes
        ---Remember to not tokenize the special tokens---
        Step 2: apply BPE merges to each pre-token in the order defined by merges
        Step 3: map each resulting token to its corresponding ID in the vocab and stop when no more merges can be applied to a pre-token
        """

        token_ids, _ = self._encode_stream_chunk(text, final=True)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            tokens, buffer = self._encode_stream_chunk(buffer, final=False)
            for tok_id in tokens:
                yield tok_id
        if buffer:
            tokens, _ = self._encode_stream_chunk(buffer, final=True)
            for tok_id in tokens:
                yield tok_id

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        out_bytes = b"".join(self.vocab[_id] for _id in ids)
        return out_bytes.decode("utf-8", errors="replace")
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab: dict[int, bytes] = {
            int(token_id): base64.b64decode(token_b64)
            for token_id, token_b64 in vocab_json.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
        merges: list[tuple[bytes, bytes]] = [
            (base64.b64decode(a), base64.b64decode(b)) for a, b in merges_json
        ]

        return cls(vocab, merges, special_tokens=special_tokens)
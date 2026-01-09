import os
import regex as re
import heapq
import time
import datetime
import json
import base64
import multiprocessing as mp
from typing import BinaryIO
from collections import defaultdict
from collections import Counter

# -----------------------------
# Multiprocessing pre-tokenization helpers
# -----------------------------

_WORKER_SPLIT_RE = None
_WORKER_PAT_RE = None


def _init_pretok_worker(special_tokens: list[str]) -> None:
    """Initializer for multiprocessing workers (sets up compiled regexes)."""
    global _WORKER_SPLIT_RE, _WORKER_PAT_RE
    # Split documents based on special tokens (same behavior as pretok_regex)
    _WORKER_SPLIT_RE = re.compile("|".join(re.escape(t) for t in special_tokens))
    # Same PAT as your serial pre-tokenization
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _WORKER_PAT_RE = re.compile(pat)


def _pretok_count_chunk(args: tuple[str, int, int]) -> Counter:
    """Worker: read [start,end) bytes from file, pre-tokenize, return Counter[bytes]."""
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")

    out: Counter = Counter()
    # IMPORTANT for memory: don't materialize a huge `docs = split(...)` list.
    # Instead, iterate segments between special tokens and tokenize each segment.
    if _WORKER_SPLIT_RE is None:
        for m in _WORKER_PAT_RE.finditer(chunk):
            out[m.group().encode("utf-8")] += 1
    else:
        last = 0
        for sm in _WORKER_SPLIT_RE.finditer(chunk):
            segment = chunk[last:sm.start()]
            for m in _WORKER_PAT_RE.finditer(segment):
                out[m.group().encode("utf-8")] += 1
            last = sm.end()
        segment = chunk[last:]
        for m in _WORKER_PAT_RE.finditer(segment):
            out[m.group().encode("utf-8")] += 1
    return out

class ReverseBytes:
    """
    Wrapper class to reverse lexicographic comparison of bytes.
    Used in min-heap to get lexicographically LARGEST pair when counts are equal.
    """
    def __init__(self, bytes_obj: bytes):
        self.bytes_obj = bytes_obj
    
    def __lt__(self, other):
        # Reverse comparison: larger bytes come first in min-heap
        return self.bytes_obj > other.bytes_obj
    
    def __eq__(self, other):
        return self.bytes_obj == other.bytes_obj
    
    def __repr__(self):
        return f"ReverseBytes({self.bytes_obj})"

class Tokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # Create logs directory if it doesn't exist
        os.makedirs('cs336_basics/logs', exist_ok=True)
        
        # Get timestamp for filenames
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def default_tok_value():
            return (0, 0)
        
        self.tokens = defaultdict(default_tok_value) # str -> (int, int) : preserves pre-token counts and indexes in a tuple
        self.vocabulary = set(bytes([i for i in range(256)]))
        self.merges = []
        self.final_vocab = {}  # token_id -> bytes mapping

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    def pretok_regex(
        self, 
        text: str, 
    ) -> list[str]:
        # Split documents based on special tokens
        docs=re.split("|".join([re.escape(sp_token) for sp_token in self.special_tokens]), text)
        # Define regex pattern for pre-tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Iterate through each document to avoid cross-document merges
        for i, doc in enumerate(docs):
            tokenized_doc = re.finditer(PAT, doc)
            for token in tokenized_doc:
                t = token.group().encode("utf-8")
                if t not in self.tokens:
                    self.tokens[t] = (1, len(self.tokens)) # (count, index)
                else:
                    # Increment count, keep same index
                    old_count, old_idx = self.tokens[t]
                    self.tokens[t] = (old_count + 1, old_idx)
        
        print(f"Total unique pre-tokens so far: {len(self.tokens)}")
        # now we print all self tokens
        # for t, (count, idx) in list(self.tokens.items()):
        #     print(f"Token: {t}, Count: {count}, Index: {idx}")

    def split_tok(
        self,
    ) -> None:
        """
        Optimized BPE algorithm with incremental pair count updates.
        
        Key optimizations:
        1. Cache pair counts and only update pairs that overlap with merged pair
        2. Use priority queue (heap) for O(log P) max pair lookup
        3. Track which words contain each pair for efficient updates
        4. Only update affected words, not all words
        
        Time Complexity: O(N×L + P log P + M×(K×L_avg + log P))
        where:
        - N = number of unique words
        - L = average word length
        - P = number of unique pairs
        - M = number of merges
        - K = average occurrences of merged pair per merge
        - L_avg = average length of affected words
        """
        # Initialize vocabulary: token_id -> bytes
        # Start with 256 base tokens (all bytes)
        vocab = {i: bytes([i]) for i in range(256)}
        next_token_id = 256
        # Add special tokens into the vocabulary as atomic tokens.
        # NOTE: self.vocab_size is interpreted as the FINAL vocabulary size cap,
        # so these special tokens consume part of the budget.
        for special_token in dict.fromkeys(self.special_tokens):
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in vocab.values():
                vocab[next_token_id] = special_bytes
                next_token_id += 1
        init_vocab_size = next_token_id
        
        # Convert each pre-token to a list of byte IDs
        # word_tokens: list of (word_bytes, count, token_id_list)
        word_tokens = []
        for token_bytes, (count, _) in self.tokens.items():
            token_id_list = list(token_bytes)  # Convert bytes to list of byte IDs
            word_tokens.append((token_bytes, count, token_id_list))
        
        # Initialize pair counts and track which words contain each pair
        # pair_counts: (token1, token2) -> count
        # pair_to_words: (token1, token2) -> set of word indices
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)
        
        # Build initial pair counts (one-time cost: O(N × L))
        for word_idx, (_, count, token_id_list) in enumerate(word_tokens):
            for i in range(len(token_id_list) - 1):
                pair = (token_id_list[i], token_id_list[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(word_idx)
        
        print(f'Initial unique pairs: {len(pair_counts)}')
        # print('pair_counts\n')
        # print(pair_counts)
        # print('pair_to_words\n')
        # print(pair_to_words)
        # print('word_tokens\n')
        # print(word_tokens)

        # Calculate target number of merges.
        # If vocab_size=500, the final vocab will have at most 500 entries total, including:
        # - 256 byte tokens
        # - all special tokens added above
        # - learned BPE merges
        if self.vocab_size < init_vocab_size:
            raise ValueError(
                f"vocab_size={self.vocab_size} is smaller than the initial vocabulary "
                f"size={init_vocab_size} (256 byte tokens + {init_vocab_size - 256} special tokens)."
            )
        target_merges = max(0, self.vocab_size - init_vocab_size)
        
        # Start timing for BPE merging
        merge_start_time = time.time()

        # NOTE: this file is a debugging log. We truncate it each run so repeated executions
        # don't append multiple runs and create confusing diffs vs the reference output.
        merges_path = f'cs336_basics/logs/merges_{self.timestamp}.txt'
        os.makedirs(os.path.dirname(merges_path), exist_ok=True)
        with open(merges_path, "w", encoding="utf-8") as _f:
            _f.write("")
        
        # Perform merges with incremental updates
        for merge_iter in range(target_merges):
            if not pair_counts:
                break  # No more pairs to merge
        
            # Build priority queue: (-count, ReverseBytes(bytes1), ReverseBytes(bytes2), pair) for max heap
            # Negative count because heapq is a min-heap
            # Tie-breaking: lexicographically LARGEST by byte values (not token IDs!)
            # ReverseBytes reverses comparison so largest comes first in min-heap

            most_freq_pair = max(pair_counts.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))[0]
            # pair_heap = []
            # for pair, count in pair_counts.items():
            #     # Get actual byte representations for tie-breaking
            #     bytes1 = vocab[pair[0]]
            #     bytes2 = vocab[pair[1]]
            #     # Use negative count for max heap, then ReverseBytes for lexicographic tie-breaking (largest first)
            #     heapq.heappush(pair_heap, (-count, ReverseBytes(bytes1), ReverseBytes(bytes2), pair))
            # print(f'Initial heap size: {len(pair_heap)}')
            # # print('Initial pair_heap\n')
            # # print(pair_heap)
            
            # # Get most frequent pair from heap (O(log P))
            # if not pair_heap:
            #     break  # No more pairs to merge
            
            # # Pop pairs until we find one that still exists in pair_counts
            # # (pairs may have been removed/updated)
            # while pair_heap:
            #     neg_count, _, _, pair = heapq.heappop(pair_heap)
            #     if pair in pair_counts and pair_counts[pair] == -neg_count:
            #         most_freq_pair = pair
            #         break
            # else:
            #     break  # No valid pairs left
            # Get the byte representations of the pair
            part1_id, part2_id = most_freq_pair
            part1_bytes = vocab[part1_id]
            part2_bytes = vocab[part2_id]
            merged_bytes = part1_bytes + part2_bytes
            merge_count = pair_counts[most_freq_pair]
            # print(f'Merge {merge_iter + 1}: {part1_bytes} {part2_bytes} with count {merge_count}')

            def _safe_decode(b: bytes) -> str:
                return b.decode("utf-8", errors="replace")

            write_to_file = (
                f'Merge {merge_iter}: "{_safe_decode(part1_bytes)}" + "{_safe_decode(part2_bytes)}" = "{_safe_decode(merged_bytes)}" ({merge_count} instances)\n'
            )
            with open(merges_path, 'a') as f:
                f.write(write_to_file)
            # Add merge to list
            self.merges.append((part1_bytes, part2_bytes))
            
            # Add new token to vocabulary
            vocab[next_token_id] = merged_bytes
            new_token_id = next_token_id
            next_token_id += 1
            
            # Get words that contain this pair (only these need updating)
            affected_words = pair_to_words[most_freq_pair].copy()
            
            def _pair_freqs(token_ids: list[int]) -> dict[tuple[int, int], int]:
                freqs = defaultdict(int)
                for pi in range(len(token_ids) - 1):
                    freqs[(token_ids[pi], token_ids[pi + 1])] += 1
                return freqs

            # Update affected words and pair counts (correctness-first: per-word diff)
            for word_idx in affected_words:
                _, count, token_id_list = word_tokens[word_idx]
                
                # Snapshot old pairs for this word
                old_pair_freqs = _pair_freqs(token_id_list)

                # Build merged token list for this word (non-overlapping, left-to-right)
                new_list = []
                i = 0
                while i < len(token_id_list):
                    if (i < len(token_id_list) - 1 and 
                        token_id_list[i] == part1_id and 
                        token_id_list[i + 1] == part2_id):
                        new_list.append(new_token_id)
                        i += 2
                    else:
                        new_list.append(token_id_list[i])
                        i += 1

                # Snapshot new pairs for this word
                new_pair_freqs = _pair_freqs(new_list)

                # Apply delta to global pair counts (weighted by word multiplicity)
                for pair, freq in old_pair_freqs.items():
                    prev = pair_counts.get(pair, 0)
                    updated = prev - (count * freq)
                    if updated <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = updated

                for pair, freq in new_pair_freqs.items():
                    pair_counts[pair] += (count * freq)
                    # Correctness-first: keep this as a superset (never remove word_idx),
                    # so we don't miss future merges for a pair that still occurs elsewhere in the word.
                    pair_to_words[pair].add(word_idx)
                
                # Update the word
                word_tokens[word_idx] = (word_tokens[word_idx][0], count, new_list)
            
            # Remove the merged pair from counts and tracking
            pair_counts.pop(most_freq_pair, None)
            pair_to_words.pop(most_freq_pair, None)
        
        # Store final vocabulary and merges
        self.final_vocab = vocab
        merge_end_time = time.time()
        merge_time = merge_end_time - merge_start_time
        
        print(f'Completed {len(self.merges)} merges in {merge_time:.2f} seconds')
        print(f'Average time per merge: {merge_time / max(len(self.merges), 1):.4f} seconds')
        # now we print all merges in format (token1, token2)
        # for i, (t1, t2) in enumerate(self.merges):
        #     print(f'Merge {i+1}: ({t1}, {t2})')
        # now we print the final vocabulary in format token_id -> bytes
        # print('Final Vocabulary:')
        # for token_id, token_bytes in self.final_vocab.items():
        #     print(f'Token ID: {token_id}, Bytes: {token_bytes}') if token_id >= 256 else None
        print(f'Final vocabulary size: {len(vocab)}')

    def bpe_tokenizer(
        self,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train BPE tokenizer on the input corpus.
        
        Returns:
            tuple: (vocab, merges)
                vocab: dict mapping token_id -> bytes
                merges: list of (token1, token2) tuples in merge order
        """
        # Start total timing
        total_start_time = time.time()
        
        # Reset state
        self.tokens = defaultdict(lambda: (0, 0))
        self.merges = []
        self.final_vocab = {}
        
        # Start timing for pre-tokenization
        pretok_start_time = time.time()
        
        # ---- Parallel pre-tokenization (multiprocessing) ----
        # We split the file into boundaries aligned on the special token, then have each worker
        # open the file and pre-tokenize its chunk. This avoids shipping huge strings between processes.
        num_workers = min(4, os.cpu_count() or 4)
        target_chunk_bytes = 16 * 1024 * 1024  # ~16MB per task (decoded string will be larger)
        with open(self.input_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            desired_num_chunks = max(num_workers, int(file_size // target_chunk_bytes) + 1)
            desired_num_chunks = min(desired_num_chunks, 256)
            boundaries = self.find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

        print(f"Found {len(boundaries)-1} chunks to process using {num_workers} processes")
        tasks = [(self.input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_pretok_worker,
            initargs=(list(self.special_tokens),),
            maxtasksperchild=1,
        ) as pool:
            for j, counter in enumerate(pool.imap_unordered(_pretok_count_chunk, tasks, chunksize=1), 1):
                # Merge counts into self.tokens (keeps your (count, index) structure)
                for tok_bytes, c in counter.items():
                    if tok_bytes not in self.tokens:
                        self.tokens[tok_bytes] = (c, len(self.tokens))
                    else:
                        old_count, old_idx = self.tokens[tok_bytes]
                        self.tokens[tok_bytes] = (old_count + c, old_idx)
                if j % 10 == 0 or j == len(tasks):
                    print(f"Processed chunks: {j}/{len(tasks)} | unique pre-tokens: {len(self.tokens)}")
        
        pretok_end_time = time.time()
        pretok_time = pretok_end_time - pretok_start_time
        print(f"\n{'='*60}")
        print(f"Pre-tokenization completed in {pretok_time:.2f} seconds")
        print(f"Total unique pre-tokens: {len(self.tokens)}")
        print(f"{'='*60}\n")
        
        # Now perform BPE merging
        self.split_tok()
        
        # Calculate total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\n{'='*60}")
        print(f"BPE Tokenization Summary:")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"  Pre-tokenization time: {pretok_time:.2f} seconds ({pretok_time/total_time*100:.1f}%)")
        print(f"  BPE merging time: {total_time - pretok_time:.2f} seconds ({(total_time - pretok_time)/total_time*100:.1f}%)")
        print(f"  Vocabulary size: {len(self.final_vocab)}")
        print(f"  Number of merges: {len(self.merges)}")
        print(f"{'='*60}\n")

        # Longest token (by byte length) in the final vocabulary
        longest_token_id, longest_token_bytes = max(
            self.final_vocab.items(),
            key=lambda kv: (len(kv[1]), kv[1]),
        )
        longest_token_len = len(longest_token_bytes)
        longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
        
        # log above information by creating a new file in logs directory and add current datetime in filename
        pathname = f'cs336_basics/logs/bpe_tokenization_summary_{self.timestamp}.txt'
        with open(pathname, 'w') as f:
            f.write(f"BPE Tokenization Summary:\n")
            f.write(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
            f.write(f"  Pre-tokenization time: {pretok_time:.2f} seconds ({pretok_time/total_time*100:.1f}%)\n")
            f.write(f"  BPE merging time: {total_time - pretok_time:.2f} seconds ({(total_time - pretok_time)/total_time*100:.1f}%)\n")
            f.write(f"  Vocabulary size: {len(self.final_vocab)}\n")
            f.write(f"  Number of merges: {len(self.merges)}\n")
            f.write(f"  Longest token length (bytes): {longest_token_len}\n")
            f.write(f"  Longest token id: {longest_token_id}\n")
            f.write(f"  Longest token (utf-8, replace): {longest_token_str}\n")
        
        # Convert vocabulary to JSON-serializable format (bytes -> base64 string)
        vocab_json = {
            str(token_id): token_bytes.decode('utf-8', errors="replace")
            for token_id, token_bytes in self.final_vocab.items()
        }
        
        # Save the final vocabulary to a json file in logs directory
        vocab_path = f'cs336_basics/logs/final_vocabulary_{self.timestamp}.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab_json, f, indent=2)
        
        print(f"Saved vocabulary to: {vocab_path}")
        return (self.final_vocab, self.merges)

if __name__ == "__main__":
    bpe_tokenizer = Tokenizer('/Users/vitthalbhandari/Code/cs336/cs-336-assignment1-llms/data/owt_train.txt', 32000, ["<|endoftext|>"])
    final_vocab, merges = bpe_tokenizer.bpe_tokenizer()
import os
import regex as re
import heapq
import time
import datetime
from typing import BinaryIO
from collections import defaultdict


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
        for special_token in self.special_tokens:
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
        
        # Build priority queue: (-count, ReverseBytes(bytes1), ReverseBytes(bytes2), pair) for max heap
        # Negative count because heapq is a min-heap
        # Tie-breaking: lexicographically LARGEST by byte values (not token IDs!)
        # ReverseBytes reverses comparison so largest comes first in min-heap
        pair_heap = []
        for pair, count in pair_counts.items():
            # Get actual byte representations for tie-breaking
            bytes1 = vocab[pair[0]]
            bytes2 = vocab[pair[1]]
            # Use negative count for max heap, then ReverseBytes for lexicographic tie-breaking (largest first)
            heapq.heappush(pair_heap, (-count, ReverseBytes(bytes1), ReverseBytes(bytes2), pair))
        print(f'Initial heap size: {len(pair_heap)}')
        # print('Initial pair_heap\n')
        # print(pair_heap)
        
        # Calculate target number of merges
        target_merges = self.vocab_size - init_vocab_size  # We start with 256 base tokens and all special tokens
        
        # Start timing for BPE merging
        merge_start_time = time.time()
        
        # Perform merges with incremental updates
        for merge_iter in range(target_merges):
            # Get most frequent pair from heap (O(log P))
            if not pair_heap:
                break  # No more pairs to merge
            
            # Pop pairs until we find one that still exists in pair_counts
            # (pairs may have been removed/updated)
            while pair_heap:
                neg_count, _, _, pair = heapq.heappop(pair_heap)
                if pair in pair_counts and pair_counts[pair] == -neg_count:
                    most_freq_pair = pair
                    break
            else:
                break  # No valid pairs left
            # print(f'Merge {merge_iter + 1}: Merging pair {most_freq_pair} with count {pair_counts[most_freq_pair]}')
            # Get the byte representations of the pair
            part1_id, part2_id = most_freq_pair
            part1_bytes = vocab[part1_id]
            part2_bytes = vocab[part2_id]
            merged_bytes = part1_bytes + part2_bytes
            
            # Add merge to list
            self.merges.append((part1_bytes, part2_bytes))
            
            # Add new token to vocabulary
            vocab[next_token_id] = merged_bytes
            new_token_id = next_token_id
            next_token_id += 1
            
            # Get words that contain this pair (only these need updating)
            affected_words = pair_to_words[most_freq_pair].copy()
            
            # Update affected words and pair counts incrementally
            for word_idx in affected_words:
                _, count, token_id_list = word_tokens[word_idx]
                
                # Find all occurrences of the pair in this word
                new_list = []
                i = 0
                while i < len(token_id_list):
                    if (i < len(token_id_list) - 1 and 
                        token_id_list[i] == part1_id and 
                        token_id_list[i + 1] == part2_id):
                        # Found the pair - need to update adjacent pairs
                        
                        # Remove old pairs: (before, part1) and (part2, after)
                        if i > 0:
                            before_pair = (token_id_list[i - 1], part1_id)
                            pair_counts[before_pair] -= count
                            if pair_counts[before_pair] <= 0:
                                del pair_counts[before_pair]
                            pair_to_words[before_pair].discard(word_idx)
                            # Re-add to heap with updated count (only if count > 0)
                            if before_pair in pair_counts:
                                before_bytes1 = vocab[before_pair[0]]
                                before_bytes2 = vocab[before_pair[1]]
                                heapq.heappush(pair_heap, (-pair_counts[before_pair], 
                                                          ReverseBytes(before_bytes1), 
                                                          ReverseBytes(before_bytes2), 
                                                          before_pair))
                        
                        if i + 2 < len(token_id_list):
                            after_pair = (part2_id, token_id_list[i + 2])
                            pair_counts[after_pair] -= count
                            if pair_counts[after_pair] <= 0:
                                del pair_counts[after_pair]
                            pair_to_words[after_pair].discard(word_idx)
                            # Re-add to heap with updated count (only if count > 0)
                            if after_pair in pair_counts:
                                after_bytes1 = vocab[after_pair[0]]
                                after_bytes2 = vocab[after_pair[1]]
                                heapq.heappush(pair_heap, (-pair_counts[after_pair],
                                                          ReverseBytes(after_bytes1), 
                                                          ReverseBytes(after_bytes2), 
                                                          after_pair))
                        
                        # Add new token
                        new_list.append(new_token_id)
                        
                        # Add new pairs: (before, new_token) and (new_token, after)
                        if i > 0:
                            new_before_pair = (token_id_list[i - 1], new_token_id)
                            pair_counts[new_before_pair] += count
                            pair_to_words[new_before_pair].add(word_idx)
                            new_before_bytes1 = vocab[new_before_pair[0]]
                            new_before_bytes2 = vocab[new_before_pair[1]]
                            heapq.heappush(pair_heap, (-pair_counts[new_before_pair],
                                                      ReverseBytes(new_before_bytes1), 
                                                      ReverseBytes(new_before_bytes2), 
                                                      new_before_pair))
                        
                        if i + 2 < len(token_id_list):
                            new_after_pair = (new_token_id, token_id_list[i + 2])
                            pair_counts[new_after_pair] += count
                            pair_to_words[new_after_pair].add(word_idx)
                            new_after_bytes1 = vocab[new_after_pair[0]]
                            new_after_bytes2 = vocab[new_after_pair[1]]
                            heapq.heappush(pair_heap, (-pair_counts[new_after_pair],
                                                      ReverseBytes(new_after_bytes1), 
                                                      ReverseBytes(new_after_bytes2), 
                                                      new_after_pair))
                        
                        i += 2
                    else:
                        new_list.append(token_id_list[i])
                        i += 1
                
                # Update the word
                word_tokens[word_idx] = (word_tokens[word_idx][0], count, new_list)
            
            # Remove the merged pair from counts and tracking
            del pair_counts[most_freq_pair]
            del pair_to_words[most_freq_pair]
        
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
        
        ## Open file for chunking
        with open(self.input_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            
            print(f"Found {len(boundaries)-1} chunks to process")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for j, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                if j % 10 == 0:
                    print(f"Processing chunk {j}/{len(boundaries)-1}")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                self.pretok_regex(chunk)
        
        pretok_end_time = time.time()
        pretok_time = pretok_end_time - pretok_start_time
        print(f"\n{'='*60}")
        print(f"Pre-tokenization completed in {pretok_time:.2f} seconds")
        print(f"Total unique pre-tokens: {len(self.tokens)}")
        print(f"{'='*60}\n")
        
        # Now perform BPE merging
        self.split_tok()
        
        # Add special tokens to vocabulary if not already present
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.final_vocab.values():
                # Find next available token ID
                max_id = max(self.final_vocab.keys()) if self.final_vocab else 255
                self.final_vocab[max_id + 1] = special_bytes
        
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

        # log above information by creating a new file in logs directory and add current datetime in filename
        with open(f'cs336_basics/logs/bpe_tokenization_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            f.write(f"BPE Tokenization Summary:\n")
            f.write(f"  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
            f.write(f"  Pre-tokenization time: {pretok_time:.2f} seconds ({pretok_time/total_time*100:.1f}%)\n")
            f.write(f"  BPE merging time: {total_time - pretok_time:.2f} seconds ({(total_time - pretok_time)/total_time*100:.1f}%)\n")
            f.write(f"  Vocabulary size: {len(self.final_vocab)}\n")
            f.write(f"  Number of merges: {len(self.merges)}\n")
        
        return (self.final_vocab, self.merges)

# if __name__ == "__main__":
#     bpe_tokenizer = Tokenizer('data/TinyStoriesV2-GPT4-valid.txt', 2000, ["<|endoftext|>"])
#     final_vocab, merges = bpe_tokenizer.bpe_tokenizer()
import os
import regex as re
from typing import BinaryIO
from collections import defaultdict

class Tokenizer:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokens = defaultdict(int)
        self.vocabulary = set(bytes([i for i in range(256)]))
        self.merges = []
        print(self.vocabulary)

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
    
    def pretok_regex(self, text: str, special_tokens: list[str]) -> list[str]:
        # Split documents based on special tokens
        docs=re.split("|".join([re.escape(sp_token) for sp_token in special_tokens]), text)
        # Define regex pattern for pre-tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Iterate through each document to avoid cross-document merges
        i=0
        for doc in docs:
            print(f"Document {i}:")
            # print(doc)
            i=i+1
            tokenized_doc = re.finditer(PAT, doc)
            for token in tokenized_doc:
                self.tokens[token.group().encode("utf-8")] += 1
            print(self.tokens)

    def split_tok(self) -> list[str]:
        def default_tuple_value():
            return (0, [])
        temp_vocab = defaultdict(default_tuple_value)
        for token, count in self.tokens.items():
            for i in range(len(token) - 1):
                before = token[i-1] if i-1 >= 0 else None
                after = token[i + 2] if i + 2 < len(token) else None
                currElement = (token[i], token[i+1])
                newCount = temp_vocab[currElement][0] + count
                newNeighbors = temp_vocab[currElement][1] + [(before, after)]
                temp_vocab[currElement] = (newCount, newNeighbors)
        
        temp_vocab_sorted = {k: v for k, v in sorted(temp_vocab.items(), key=lambda item: (item[1][0], item[0][0], item[0][1]), reverse=True)}
        print(len(temp_vocab_sorted))
        print(temp_vocab_sorted) 
        # now we want to begin merging until we obtain 10k merges
        while len(self.temp_vocab_sorted) < 10000:
            # get the highest count pair
            most_freq_pair = next(iter(temp_vocab_sorted))

            # put this in the merges list
            self.merges.append(most_freq_pair)

            # iterate through this pair's neighbors and update the tokens
            neighbors = temp_vocab_sorted[most_freq_pair][1]
            for before, after in neighbors:

                # update the keys of before and after token tuples
                if before is not None:
                    before_pair = (before, most_freq_pair[0])
                    new_before_pair = (before, b"".join(most_freq_pair))
                    if before_pair in temp_vocab_sorted:
                        temp_vocab_sorted[new_before_pair] = temp_vocab_sorted.pop(before_pair)
                        # after updating the key, we need to update the neighbors list
                        updated_neighbors = []
                        for n_before, n_after in temp_vocab_sorted[new_before_pair][1]:
                            if n_after == most_freq_pair[1]:
                                updated_neighbors.append((n_before, after))
                            else:
                                updated_neighbors.append((n_before, n_after))
                
                if after is not None:
                    after_pair = (most_freq_pair[1], after)
                    new_after_pair = (b"".join(most_freq_pair), after)
                    if after_pair in temp_vocab_sorted:
                        temp_vocab_sorted[new_after_pair] = temp_vocab_sorted.pop(after_pair)
                        # after updating the key, we need to update the neighbors list
                        updated_neighbors = []
                        for n_before, n_after in temp_vocab_sorted[new_after_pair][1]:
                            if n_before == most_freq_pair[0]:
                                updated_neighbors.append((before, n_after))
                            else:
                                updated_neighbors.append((n_before, n_after))

        print(len(temp_vocab_sorted))
        print(temp_vocab_sorted)

    def bpe_tokenizer(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> {dict[int, bytes], list[tuple[bytes, bytes]]}:
    
        ## Open file for chunking
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            j=1
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                print(f"Chunk {j}:")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                if j==1:
                    self.pretok_regex(chunk, special_tokens)
                    self.split_tok()
                j=j+1

        return {}  # Placeholder return

if __name__ == "__main__":
    bpe_tokenizer = Tokenizer("path/to/corpus.txt")
    resp = bpe_tokenizer.bpe_tokenizer('data/TinyStoriesMini.txt', 10000, ["<|endoftext|>"])
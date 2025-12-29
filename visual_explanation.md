# Visual Explanation of `pretokenization_example.py`

## The Problem
We want to split a large file into chunks for **parallel processing**, but we need to ensure that special tokens (`<|endoftext|>`) are never split across chunks.

## Visual Example

```
File content: "Text1<|endoftext|>Text2<|endoftext|>Text3"
File size: 41 bytes
Special tokens at positions: 5, 23
```

### Step 1: Initial Rough Guess
Want **2 chunks**, so:
- Chunk size ≈ 41 / 2 = 20 bytes
- Initial boundaries: [0, 20, 41]

```
Byte:    0               20              41
         |---------------|----------------|
         |   Chunk 1     |   Chunk 2      |
Content: |Text1<|endoftext|>Te|xt2<|endoftext|>Text3|
                                      ↑
                        Boundary at position 20 splits text mid-token!
```

**Problem:** The boundary at position 20 cuts through `Text2<|endoftext|>`, which could break tokenization!

---

### Step 2: Algorithm Alignment

**Goal:** Align the boundary at position 20 to the nearest `<|endoftext|>` token.

**What the code does:**

1. Start at initial boundary position (20)
2. Read ahead in small chunks (4KB at a time)
3. Look for the special token `<|endoftext|>`
4. When found, set the boundary to that position

**In our example:**
```
Position 20: Looking forward...
  At bytes 20-35: "xt2<|endoftext|>"
  ✓ Found "<|endoftext|>" at offset 3 within this chunk
  New boundary = 20 + 3 = 23
```

---

### Step 3: Final Result

**Aligned boundaries: [0, 23, 41]**

```
Byte:    0               23                         41
         |---------------|--------------------------|
         |   Chunk 1     |      Chunk 2             |
Content: |Text1<|endoftext|>Text2|<|endoftext|>Text3|
                      ↑          ↑               ↑
                   Token 1   Boundary        Token 2
                              (aligned!)
```

**Result:**
- Chunk 1: `"Text1<|endoftext|>Text2"` (23 bytes)
- Chunk 2: `"<|endoftext|>Text3"` (18 bytes)

✓ Both chunks end at complete special tokens
✓ Can be processed independently in parallel
✓ No partial tokens to handle

---

## Code Breakdown by Lines

### Lines 5-9: Function signature
```python
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
```
- `file`: The file to chunk (opened in binary mode)
- `desired_num_chunks`: How many chunks you want (e.g., 4)
- `split_special_token`: The token that must NOT be split (e.g., `b"<|endoftext|>"`)
- Returns: List of byte positions where chunks start/end

### Lines 17-19: Get file size
```python
file.seek(0, os.SEEK_END)  # Jump to end
file_size = file.tell()       # Current position = file size
file.seek(0)                  # Reset to beginning
```

### Lines 21-26: Initial boundary guesses
```python
chunk_size = file_size // desired_num_chunks      # Approx size per chunk
chunk_boundaries = [0, chunk_size, 2*chunk_size, ..., file_size]
```

### Lines 30-46: The magic - aligning boundaries
```python
for each boundary (except first and last):
    position = initial_guess
    while True:
        read 4KB chunk
        if EOF: set boundary to end of file
        if found special token:
            boundary = position + offset of token
            break
        else:
            position += 4KB  # Keep scanning forward
```

### Line 49: Return unique, sorted boundaries
```python
return sorted(set(chunk_boundaries))
```

---

## Why This Matters for Tokenization

When pre-tokenizing (converting text to tokens), you need to count tokens. If you split the file at arbitrary positions:

**BAD:**
```
Chunk 1: "...partial text<|en"
Chunk 2: "doftext|>..."
```
- Chunk 1 has incomplete `<|en` 
- Chunk 2 has incomplete `doftext|>`
- These aren't valid tokens!

**GOOD (with aligned chunks):**
```
Chunk 1: "...text<|endoftext|>"
Chunk 2: "<|endoftext|>more text..."
```
- Each chunk has complete tokens
- Can process independently
- Can combine token counts safely

---

## Real-World Usage

```python
with open("huge_file.txt", "rb") as f:
    num_processes = 8  # Use 8 workers
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # Split work across processes
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        # Send to worker process for tokenization
        # Each worker counts tokens independently
```

**Key insight:** All chunks end at complete special tokens, so:
1. No tokens are split across chunks
2. Results can be safely combined
3. Parallel processing is safe


from pathlib import Path
import codecs
import json
import numpy as np
from tokenize_corpus import Tokenizer


def iter_text_chunks(path: Path, chunk_size: int = 1 << 20):
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield decoder.decode(data), len(data)
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail, 0


def encode_file_to_uint16_bin(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_size: int = 1 << 20,
) -> dict[str, float]:
    # Single pass: stream tokens to a raw uint16 binary file
    token_count = 0
    out = open(output_path, "wb")
    bytes_read = 0
    last_report = 0
    report_every = 256 * 1024 * 1024  # 256MB
    for chunk, nbytes in iter_text_chunks(input_path, chunk_size=chunk_size):
        bytes_read += nbytes
        for token_id in tokenizer.encode_iterable([chunk]):
            if token_id > np.iinfo(np.uint16).max:
                raise ValueError(f"Token id {token_id} exceeds uint16 max.")
            out.write(np.uint16(token_id).tobytes())
            token_count += 1
        if bytes_read - last_report >= report_every:
            print(
                f"  progress: {bytes_read / (1024**2):.1f} MB read, {token_count} tokens",
                flush=True,
            )
            last_report = bytes_read
    out.flush()
    out.close()

    return {
        "tokens": float(token_count),
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent
    data_dir = repo_root / "data"
    logs_dir = base_dir.parent / "logs"
    output_dir = data_dir / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)
    delimiter = "<|endoftext|>"

    tinystories_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(logs_dir / "final_vocabulary_20260122_101516_tinystories_train.json"),
        merges_filepath=str(logs_dir / "merges_20260122_101516_tinystories_train.json"),
        special_tokens=[delimiter],
    )

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(logs_dir / "final_vocabulary_20260122_102544_owt_train.json"),
        merges_filepath=str(logs_dir / "merges_20260122_102544_owt_train.json"),
        special_tokens=[delimiter],
    )

    datasets = [
        ("tinystories_train", data_dir / "TinyStoriesV2-GPT4-train.txt", tinystories_tokenizer),
        ("tinystories_valid", data_dir / "TinyStoriesV2-GPT4-valid.txt", tinystories_tokenizer),
        ("owt_train", data_dir / "owt_train.txt", owt_tokenizer),
        ("owt_valid", data_dir / "owt_valid.txt", owt_tokenizer),
    ]

    for name, path, tok in datasets:
        out_path = output_dir / f"{name}_uint16.bin"
        print(f"Encoding {name} from {path} -> {out_path}", flush=True)
        stats = encode_file_to_uint16_bin(tok, path, out_path)
        meta_path = output_dir / f"{name}_uint16.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "dtype": "uint16",
                    "tokens": int(stats["tokens"]),
                    "source": str(path),
                },
                f,
                indent=2,
            )
        print(
            f"{name}: tokens={stats['tokens']:.0f}, out={out_path}, meta={meta_path}"
        )

    # Loader snippet (reads raw uint16 binary into a NumPy array)
    # Example usage:
    # tokens = load_uint16_bin(output_dir / "owt_train_uint16.bin")
    def load_uint16_bin(path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.uint16)
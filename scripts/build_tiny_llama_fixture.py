#!/usr/bin/env python3
"""Build a tiny local Llama checkpoint for smoke-testing the RYS pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny random Llama fixture")
    parser.add_argument("--output-dir", required=True, help="Where to save the fixture")
    parser.add_argument(
        "--tokenizer",
        default="hf-internal-testing/llama-tokenizer",
        help="Tokenizer repo to reuse for the tiny fixture",
    )
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = LlamaForCausalLM(config)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Tiny fixture written to {output_dir}")


if __name__ == "__main__":
    main()

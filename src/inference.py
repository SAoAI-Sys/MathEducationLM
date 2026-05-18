"""
inference.py
============
Gemma-4-E4B-it + 学習済み LoRA adapter を使ってチャット形式で推論する。

使い方:
    uv run inference.py
    uv run inference.py --adapter ./output/gemma4-math-final
    uv run inference.py --merged  ./output/gemma4-math-merged  # マージ済みの場合
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ============================================================
# デフォルト設定
# ============================================================

BASE_MODEL = "google/gemma-4-E4B-it"
ADAPTER_PATH = "./output/gemma4-math-final"

SYSTEM_PROMPT = (
    "あなたは数学の専門家です。"
    "与えられた問題をステップバイステップで丁寧に解いてください。"
    "最終的な答えは \\boxed{} で囲んでください。"
)

GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    temperature=0.1,  # 数学は低温でほぼ greedy に
    do_sample=True,
    repetition_penalty=1.1,
)


# ============================================================
# モデルロード
# ============================================================


def load_model_and_tokenizer(adapter_path: str | None, merged_path: str | None):
    """
    adapter_path : LoRA adapter ディレクトリ (adapter_config.json が入っている場所)
    merged_path  : merge_and_unload 済みのディレクトリ (どちらか一方を指定)
    """
    if merged_path:
        # マージ済みモデルをそのままロード
        print(f"マージ済みモデルをロード: {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_path)

    else:
        # ベース + LoRA adapter をロード
        from peft import PeftModel

        print(f"ベースモデルをロード: {BASE_MODEL}")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        print(f"LoRA adapter をロード: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ============================================================
# 推論
# ============================================================


def build_input_ids(
    tokenizer,
    history: list[dict],
    device: str,
) -> torch.Tensor:
    """
    history: [{"role": "user"|"assistant", "content": str}, ...]
    システムプロンプトは先頭の user メッセージに埋め込む。
    """
    # Gemma 4 は system ロールを持たないため、最初の user に折り込む
    messages = []
    for i, msg in enumerate(history):
        if i == 0 and msg["role"] == "user":
            content = f"{SYSTEM_PROMPT}\n\n{msg['content']}"
            messages.append({"role": "user", "content": content})
        else:
            messages.append(msg)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    history: list[dict],
    stream: bool = True,
) -> str:
    device = next(model.parameters()).device
    input_ids = build_input_ids(tokenizer, history, str(device))

    streamer = (
        TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if stream
        else None
    )

    output_ids = model.generate(
        input_ids,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **GENERATION_CONFIG,
    )

    # 入力トークン部分を除いた生成部分のみデコード
    new_tokens = output_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# チャットループ
# ============================================================


def chat_loop(model, tokenizer) -> None:
    history: list[dict] = []

    print("\n" + "=" * 60)
    print("  Gemma-4-E4B 数学特化モデル  (終了: exit / quit / Ctrl-C)")
    print("  history クリア: /clear")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("終了します。")
            break

        if user_input == "/clear":
            history.clear()
            print("[会話履歴をクリアしました]\n")
            continue

        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        response = generate(model, tokenizer, history)
        print()  # ストリーム後の改行

        history.append({"role": "assistant", "content": response})
        print()


# ============================================================
# エントリーポイント
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma-4-E4B 数学特化モデル 推論")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "-a",
        "--adapter",
        default=ADAPTER_PATH,
        metavar="PATH",
        help=f"LoRA adapter ディレクトリ (デフォルト: {ADAPTER_PATH})",
    )
    group.add_argument(
        "-m",
        "--merged",
        default=None,
        metavar="PATH",
        help="merge_and_unload 済みモデルディレクトリ",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print(
            "警告: CUDA が利用できません。CPU で実行します（非常に遅い）。",
            file=sys.stderr,
        )

    model, tokenizer = load_model_and_tokenizer(
        adapter_path=args.adapter,
        merged_path=args.merged,
    )

    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()

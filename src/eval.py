"""
eval_gsm8k.py
=============
openai/gsm8k の test split で Gemma-4-E4B-it (ベース) と
マージ済みファインチューニングモデルの精度を比較する。

スコア:  正解数 / 問題総数 × 100 [%]
表示:    matplotlib による棒グラフ (ベース vs マージ)

使い方:
    uv run eval_gsm8k.py
    uv run eval_gsm8k.py --max-samples 200       # サンプル数を絞る (デフォルト: 全件)
    uv run eval_gsm8k.py --batch-size 8          # バッチサイズ
    uv run eval_gsm8k.py --merged-path ./output/gemma4-math-merged
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================
# 定数
# ============================================================

BASE_MODEL = "google/gemma-4-E4B-it"
MERGED_PATH_DEFAULT = "./output/gemma4-math-merged"

SYSTEM_PROMPT = (
    "あなたは数学の専門家です。"
    "与えられた問題をステップバイステップで丁寧に解いてください。"
    "最終的な答えは \\boxed{} で囲んでください。"
)

MAX_NEW_TOKENS = 512


# ============================================================
# 解答抽出
# ============================================================

# 数値パターン: 整数, 小数, 分数, カンマ区切り (1,000) 対応
_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_GSM8K_GT_RE = re.compile(r"####\s*([\-\d\.,]+)")


def _normalize_number(s: str) -> Optional[float]:
    """文字列を float に変換。失敗時 None。"""
    if s is None:
        return None
    s = s.strip().replace(",", "").rstrip(".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_gt_answer(gt_text: str) -> Optional[float]:
    """GSM8K の正解 (### N の形式) を抽出。"""
    m = _GSM8K_GT_RE.search(gt_text)
    if m:
        return _normalize_number(m.group(1))
    return None


def extract_pred_answer(pred_text: str) -> Optional[float]:
    """
    モデル出力から数値解答を抽出する。
    優先順位:
      1. \boxed{...} の中身
      2. 出力末尾の数値
    """
    # 1. \boxed{} を探す (複数あれば最後)
    boxed = _BOXED_RE.findall(pred_text)
    if boxed:
        for candidate in reversed(boxed):
            inner = _NUMBER_RE.findall(candidate)
            if inner:
                val = _normalize_number(inner[-1])
                if val is not None:
                    return val

    # 2. 末尾の数値 (出力全体の最後の数値)
    numbers = _NUMBER_RE.findall(pred_text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def is_correct(pred: Optional[float], gt: Optional[float], tol: float = 1e-3) -> bool:
    """予測値と正解の数値一致判定。"""
    if pred is None or gt is None:
        return False
    return abs(pred - gt) <= tol


# ============================================================
# プロンプト構築
# ============================================================

def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    """各問題をチャットテンプレートに変換。"""
    prompts = []
    for q in questions:
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n問題:\n{q}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


# ============================================================
# バッチ推論
# ============================================================

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """バッチ推論。プロンプトを除いた生成テキストのみを返す。"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy で再現性確保
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # プロンプト部分を除いた生成のみデコード
    input_lengths = inputs.input_ids.shape[1]
    generated = output_ids[:, input_lengths:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ============================================================
# 評価ループ
# ============================================================

def evaluate_model(
    model_path: str,
    questions: list[str],
    gt_answers: list[float],
    batch_size: int,
    label: str,
) -> tuple[float, int, int]:
    """
    Returns:
        (accuracy[%], correct_count, total_count)
    """
    print(f"\n{'=' * 60}")
    print(f"  評価中: {label}  ({model_path})")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only は left padding

    # 推論時も 4bit 量子化でロード。
    # Gemma-4-E4B の BF16 重みは 15.1GB あり、RTX 4500 Ada (24GB) では
    # KV キャッシュ等の overhead を含めると bf16 のまま評価すると OOM になる。
    # NF4 量子化により約 5GB まで削減し、2モデルの順次評価を安全に行える。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()

    prompts = build_prompts(tokenizer, questions)

    correct = 0
    total = len(prompts)
    pbar = tqdm(range(0, total, batch_size), desc=label)
    for i in pbar:
        batch_prompts = prompts[i : i + batch_size]
        batch_gts = gt_answers[i : i + batch_size]

        try:
            outputs = generate_batch(model, tokenizer, batch_prompts, MAX_NEW_TOKENS)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  [OOM] batch_size を {batch_size} → {batch_size // 2} に縮小")
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            outputs = generate_batch(
                model, tokenizer, batch_prompts[:batch_size], MAX_NEW_TOKENS
            )

        for out, gt in zip(outputs, batch_gts):
            pred = extract_pred_answer(out)
            if is_correct(pred, gt):
                correct += 1

        pbar.set_postfix({"acc": f"{correct / (i + len(batch_prompts)) * 100:.2f}%"})

    accuracy = correct / total * 100
    print(f"  → {label}: {correct}/{total} = {accuracy:.2f}%")

    # メモリ解放
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return accuracy, correct, total


# ============================================================
# 描画
# ============================================================

def plot_results(
    base_acc: float,
    merged_acc: float,
    base_correct: int,
    merged_correct: int,
    total: int,
    output_path: str = "./output/gsm8k_eval.png",
) -> None:
    """棒グラフで比較表示。"""
    labels = ["Base\n(Gemma-4-E4B-it)", "Merged\n(Math Fine-tuned)"]
    values = [base_acc, merged_acc]
    correct_counts = [base_correct, merged_correct]
    colors = ["#94a3b8", "#10b981"]  # gray vs emerald

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2, width=0.6)

    # 棒の上に数値ラベル
    for bar, value, correct in zip(bars, values, correct_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.2f}%\n({correct}/{total})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy [%]", fontsize=12)
    ax.set_title(
        "GSM8K Test Accuracy: Base vs Merged Model",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # 改善幅を注釈
    delta = merged_acc - base_acc
    delta_color = "#10b981" if delta > 0 else "#ef4444"
    ax.text(
        0.5,
        0.95,
        f"Δ = {delta:+.2f} pp",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=delta_color,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=delta_color),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nグラフを保存: {output_path}")
    plt.show()


# ============================================================
# main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSM8K test での精度比較評価")
    p.add_argument("--merged-path", default=MERGED_PATH_DEFAULT, help="マージ済みモデルパス")
    p.add_argument("--max-samples", type=int, default=None, help="評価サンプル数上限 (デフォルト: 全件 1319)")
    p.add_argument("--batch-size", type=int, default=4, help="バッチサイズ")
    p.add_argument("--output", default="./output/gsm8k_eval.png", help="グラフ出力パス")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=== GSM8K test ロード ===")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    questions = ds["question"]
    gt_texts = ds["answer"]
    gt_answers = [extract_gt_answer(t) for t in gt_texts]

    n_valid_gt = sum(1 for v in gt_answers if v is not None)
    print(f"  問題数: {len(questions)} (正解抽出可能: {n_valid_gt})")

    # ----- ベースモデル評価 -----
    base_acc, base_correct, total = evaluate_model(
        BASE_MODEL, questions, gt_answers, args.batch_size, "Base"
    )

    # ----- マージ済みモデル評価 -----
    merged_acc, merged_correct, _ = evaluate_model(
        args.merged_path, questions, gt_answers, args.batch_size, "Merged"
    )

    # ----- 結果表示 -----
    print("\n" + "=" * 60)
    print("  最終結果")
    print("=" * 60)
    print(f"  Base   : {base_correct}/{total} = {base_acc:.2f}%")
    print(f"  Merged : {merged_correct}/{total} = {merged_acc:.2f}%")
    print(f"  Δ      : {merged_acc - base_acc:+.2f} pp")

    plot_results(base_acc, merged_acc, base_correct, merged_correct, total, args.output)


if __name__ == "__main__":
    main()
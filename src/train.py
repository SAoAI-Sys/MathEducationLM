"""
gemma4_math_sft.py
==================
Gemma-4-E4B-it を数学特化にファインチューニングする (v5)。

v4 → v5 の変更点 (TRL API 修正):
  - SFTTrainer の引数 tokenizer → processing_class
    (TRL の現行仕様。マルチモーダル対応のためトークナイザーとプロセッサ両対応の汎用名)
  - SFTConfig の warmup_ratio → warmup_steps
    (warmup_ratio は v5.2 で削除予定の deprecation 警告対応)

v3 → v4 の変更点 (Gemma 4 固有のクラッシュ対応):
  - target_modules を正規表現化し language_model 配下のみに限定
  - exclude_modules で vision_tower / audio_tower / multi_modal_projector を除外
  - torch_dtype → dtype に変更

v2 → v3 の変更点 (OOM 対策):
  - prepare_model_for_kbit_training の呼び出しを削除
  - 必要な処理を手動で実行
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True を起動時に設定

実行前推奨:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

# 注意: torch import より前に環境変数を設定する
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import re
from typing import Iterator, Optional

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# ============================================================
# 定数
# ============================================================

BASE_MODEL: str = "google/gemma-4-E4B-it"

DATASET_SPECS: list[tuple[str, str, str, str, str]] = [
    ("openai/gsm8k", "main", "train", "question", "answer"),
    (
        "gallifantjack/hendrycks_competition_math_N_A",
        "default",
        "train",
        "problem",
        "solution",
    ),
    ("AI-MO/NuminaMath-CoT", "default", "train", "problem", "solution"),
    ("open-r1/OpenR1-Math-220k", "default", "train", "problem", "solution"),
    ("SynthLabsAI/Big-Math-RL-Verified", "default", "train", "problem", "solution"),
]

MAX_SAMPLES_PER_DATASET: int = 5_000
HARD_THRESHOLD: float = 0.65
CURRICULUM_WARMUP_STEPS: int = 500
LORA_R: int = 16
MAX_SEQ_LENGTH: int = 2048


# ============================================================
# 難易度スコアリング
# ============================================================

_DIFFICULTY_KEYWORDS: list[tuple[str, float]] = [
    (r"\bprove\b", 0.30),
    (r"\bshow that\b", 0.25),
    (r"\bolympiad\b", 0.40),
    (r"\bamc\b", 0.20),
    (r"\baime\b", 0.35),
    (r"\bintegral\b", 0.20),
    (r"\bderivative\b", 0.15),
    (r"\bmatrix\b", 0.15),
    (r"\bmodular\b", 0.20),
    (r"\bcombinatorics\b", 0.25),
    (r"\bnumber theory\b", 0.25),
    (r"\bcongruent\b", 0.20),
    (r"\bprime\b", 0.10),
]
_KW_PATTERNS = [(re.compile(p, re.I), w) for p, w in _DIFFICULTY_KEYWORDS]


def difficulty_score(text: str, max_words: int = 512) -> float:
    text_lower = text.lower()
    word_count = len(text.split())
    length_score = min(word_count / max_words, 1.0)
    kw_score = min(sum(w for pat, w in _KW_PATTERNS if pat.search(text_lower)), 1.0)
    latex_hits = len(re.findall(r"\\[a-zA-Z]+", text))
    latex_score = min(latex_hits / 40, 1.0)
    return 0.40 * length_score + 0.35 * kw_score + 0.25 * latex_score


# ============================================================
# データセット
# ============================================================


def _load_single(
    ds_id: str,
    config: str,
    split: str,
    q_col: str,
    a_col: str,
    max_samples: int,
) -> Optional[Dataset]:
    try:
        if config == "default":
            raw = load_dataset(ds_id, split=split)
        else:
            raw = load_dataset(ds_id, config, split=split)

        if len(raw) > max_samples:
            raw = raw.shuffle(seed=42).select(range(max_samples))

        def normalize(ex: dict) -> dict:
            q = ex.get(q_col, "")
            a = ex.get(a_col, "")
            return {
                "question": q,
                "answer": a,
                "difficulty_score": difficulty_score(f"{q} {a}"),
            }

        normalized = raw.map(normalize, remove_columns=raw.column_names)
        print(f"  [OK] {ds_id}: {len(normalized)} samples")
        return normalized
    except Exception as e:
        print(f"  [SKIP] {ds_id}: {e}")
        return None


def load_math_datasets() -> Dataset:
    print("=== データセット読み込み ===")
    parts = []
    for spec in DATASET_SPECS:
        ds = _load_single(*spec, MAX_SAMPLES_PER_DATASET)
        if ds is not None:
            parts.append(ds)
    if not parts:
        raise RuntimeError("利用可能なデータセットが1件もありません")
    combined = concatenate_datasets(parts)
    print(f"\n合計: {len(combined)} samples")
    return combined


# ============================================================
# チャットテンプレート
# ============================================================

_SYSTEM_PROMPT = (
    "あなたは数学の専門家です。"
    "与えられた問題をステップバイステップで丁寧に解いてください。"
    "最終的な答えは \\boxed{} で囲んでください。"
)


def apply_chat_template(dataset: Dataset, tokenizer) -> Dataset:
    def _convert(ex: dict) -> dict:
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{_SYSTEM_PROMPT}\n\n問題:\n{ex['question']}",
                },
            ],
            "completion": [
                {"role": "assistant", "content": ex["answer"]},
            ],
            "difficulty_score": ex["difficulty_score"],
        }

    return dataset.map(_convert, remove_columns=["question", "answer"])


# ============================================================
# カリキュラムサンプラー
# ============================================================


class CurriculumSampler(Sampler):
    def __init__(
        self,
        scores: list[float],
        hard_threshold: float = HARD_THRESHOLD,
        warmup_steps: int = CURRICULUM_WARMUP_STEPS,
        seed: int = 42,
    ) -> None:
        self.scores = scores
        self.hard_threshold = hard_threshold
        self.warmup_steps = warmup_steps
        self.current_step: int = 0
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        self._easy: list[int] = [i for i, s in enumerate(scores) if s < hard_threshold]
        self._all: list[int] = list(range(len(scores)))

        if len(self._easy) == 0:
            raise ValueError(f"hard_threshold={hard_threshold} が高すぎます")

        n_hard = len(self._all) - len(self._easy)
        print(f"  easy: {len(self._easy)}, hard: {n_hard}, 合計: {len(self._all)}")

    def _shuffle_pool_to_size(self, pool: list[int], target_size: int) -> list[int]:
        result: list[int] = []
        while len(result) < target_size:
            perm = torch.randperm(len(pool), generator=self._rng).tolist()
            result.extend(pool[i] for i in perm)
        return result[:target_size]

    def __iter__(self) -> Iterator[int]:
        target_size = len(self._all)
        pool = self._easy if self.current_step < self.warmup_steps else self._all
        return iter(self._shuffle_pool_to_size(pool, target_size))

    def __len__(self) -> int:
        return len(self._all)


class CurriculumCallback(TrainerCallback):
    def __init__(self, sampler: CurriculumSampler) -> None:
        self.sampler = sampler
        self._notified = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.sampler.current_step = state.global_step
        if not self._notified and state.global_step >= self.sampler.warmup_steps:
            self._notified = True
            print(
                f"\n[Curriculum] step={state.global_step}: "
                "次エポックから hard サンプルを導入します。"
            )


class CurriculumSFTTrainer(SFTTrainer):
    def __init__(self, *args, curriculum_sampler: CurriculumSampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._curriculum_sampler = curriculum_sampler

    def _get_train_sampler(self, *args, **kwargs) -> Sampler:
        return self._curriculum_sampler


# ============================================================
# モデル初期化 (v3: prepare_model_for_kbit_training を使わない)
# ============================================================


def build_model_and_tokenizer(model_id: str):
    """
    QLoRA 用のモデル初期化。

    prepare_model_for_kbit_training を使わない理由:
      この関数は非量子化層 (LayerNorm / Embedding / PLE) を fp32 に昇格させる。
      Gemma 4 E4B は Per-Layer Embeddings (PLE) を搭載しており、
      その昇格時のメモリスパイクが OOM の直接原因となる。

      LoRA のみの学習では fp32 昇格は必須ではない (数値安定性のための保険)。
      代わりに以下の処理だけを手動で行う:
        - 全パラメータの requires_grad=False (get_peft_model が後で adapter のみ True にする)
        - gradient_checkpointing_enable
        - enable_input_require_grads (量子化モデル + GC で必要)
    """
    print("\n=== モデル・トークナイザー初期化 ===")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
        dtype=torch.bfloat16,  # transformers 4.x 以降は torch_dtype より dtype が推奨
    )

    # ロード後のメモリ断片化を解消
    gc.collect()
    torch.cuda.empty_cache()

    # ----- prepare_model_for_kbit_training の代替処理 -----
    # 1. 全パラメータの勾配を無効化 (LoRA adapter のみ学習する)
    for param in model.parameters():
        param.requires_grad = False

    # 2. Gradient Checkpointing 有効化
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 3. 量子化モデル + GC の組み合わせで必要な入力勾配の有効化
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # ------------------------------------------------------

    # ----- LoRA 設定 (Gemma 4 対応) -----
    # Gemma 4 は vision/audio エンコーダに Gemma4ClippableLinear ラッパーを使用しており、
    # これは nn.Linear ではなく nn.Module を継承しているため PEFT の型チェックで弾かれる。
    #
    # 単純に target_modules=["q_proj", ...] を渡すと PEFT がモデル全体を走査して
    # マルチモーダル側の ClippableLinear で ValueError を出す。
    #
    # 対策:
    #   1. target_modules を正規表現にして language_model 配下のみに限定
    #   2. exclude_modules で vision/audio 系を明示的に除外
    #
    # 数学特化なのでテキストデコーダのみ学習で十分。
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
        exclude_modules=["vision_tower", "audio_tower", "multi_modal_projector"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer


# ============================================================
# SFT 設定
# ============================================================


def build_sft_config(total_train_steps: int) -> SFTConfig:
    """
    Args:
        total_train_steps: 学習全体のステップ数
            (warmup_steps の動的計算に使用)
    """
    warmup_steps = max(int(total_train_steps * 0.05), 100)

    return SFTConfig(
        output_dir="./output/gemma4-math",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,  # warmup_ratio は v5.2 で削除予定のため warmup_steps を使用
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        max_length=MAX_SEQ_LENGTH,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_drop_last=True,
    )


# ============================================================
# main
# ============================================================


def main() -> None:
    dataset = load_math_datasets()
    model, tokenizer = build_model_and_tokenizer(BASE_MODEL)

    print("\n=== チャットテンプレート適用 ===")
    dataset = apply_chat_template(dataset, tokenizer)

    print("\n=== カリキュラムサンプラー構築 ===")
    scores: list[float] = dataset["difficulty_score"]
    curriculum_sampler = CurriculumSampler(
        scores=scores,
        hard_threshold=HARD_THRESHOLD,
        warmup_steps=CURRICULUM_WARMUP_STEPS,
    )
    curriculum_callback = CurriculumCallback(curriculum_sampler)

    # warmup_steps 計算用の総ステップ数 (per_device_batch=1, grad_accum=8, epochs=3)
    effective_batch_size = 1 * 8
    total_train_steps = (len(dataset) // effective_batch_size) * 3
    print(f"  推定総ステップ数: {total_train_steps}")

    sft_config = build_sft_config(total_train_steps)

    trainer = CurriculumSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,  # SFTTrainer は tokenizer ではなく processing_class
        curriculum_sampler=curriculum_sampler,
        callbacks=[curriculum_callback],
    )

    print("\n=== 学習開始 ===")
    print(f"  LR スケジューラ : cosine annealing")
    print(f"  ウォームアップ : {CURRICULUM_WARMUP_STEPS} steps は easy のみ")
    print(f"  hard しきい値  : {HARD_THRESHOLD}")
    trainer.train()

    trainer.save_model("./output/gemma4-math-final")
    tokenizer.save_pretrained("./output/gemma4-math-final")
    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(
    base_path: str, adapter_path: str, save_path: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    ベースモデルとSFT学習済みモデルをマージする関数

    Parameters
    ----------
    base_path : str
        マージを行うベースとなるモデル
    adapter_path : str
        マージを行うLoRAモデル
    save_path : str
        保存先のパス
    """
    # ベースモデルを bf16 でロード（GGUF 変換のために 4bit は使わない）
    print("ベースモデルをロード中...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # LoRA adapter をロードしてマージ
    print("adapter をマージ中...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # ローカルに保存（GGUF 変換用）
    print(f"ローカルに保存中: {save_path}")
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer


def uploader(
    hf_repo: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> None:
    """
    Hugging Faceにアップロードする関数

    Parameters
    ----------
    hf_repo : str
        Hugging Faceのリポジトリ
    model : transformers.AutoModelForCausalLM
        LoRAモデル本体
    tokenizer : transformers.AutoTokenizer
        LoRAモデルのトークナイザー
    """
    # HF Hub にプッシュ
    print("HF Hub にプッシュ中...")
    model.push_to_hub(HF_REPO, private=True)
    tokenizer.push_to_hub(HF_REPO, private=True)


if __name__ == "__main__":
    # 引数処理
    parser = argparse.ArgumentParser(description="学習後のモデルをマージする")
    parser.add_argument(
        "-a", "--adapter-path", required=True, help="LoRA学習後のモデルパス"
    )
    parser.add_argument(
        "-b",
        "--base-model",
        required=True,
        help="ベースモデルのローカルパス又はHuggingFaceのリポジトリID",
    )
    parser.add_argument(
        "-r",
        "--hf-repo",
        required=True,
        help="HuggingFaceにプッシュする先のリポジトリID",
    )
    parser.add_argument(
        "-p",
        "--merged-path",
        required=True,
        help="マージしたモデルを保存するローカルパス",
    )

    args = parser.parse_args()

    model, tokenizer = merge(
        base_path=args.base_model,
        adapter_path=args.adapter_path,
        save_path=args.merged_path,
    )
    uploader(hf_repo=args.hf_repo, model=model, tokenizer=tokenizer)

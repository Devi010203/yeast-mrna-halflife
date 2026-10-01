"""Validate the installed RNA-FM bundle without starting training."""

import argparse

import torch
from multimolecule import RnaFmModel, RnaTokenizer

from main import Config, validate_pretrained_directory


def check_model(model_path):
    model_dir = validate_pretrained_directory(model_path)
    tokenizer = RnaTokenizer.from_pretrained(model_dir, local_files_only=True)
    model, loading_info = RnaFmModel.from_pretrained(
        model_dir, local_files_only=True, output_loading_info=True
    )
    print("Loading information:", loading_info)
    # The original checkpoint has no pooler. main.py consumes last_hidden_state,
    # which does not depend on this optional sequence-pooling projection.
    unused_pooler = {"pooler.dense.weight", "pooler.dense.bias"}
    missing_backbone = set(loading_info.get("missing_keys", [])) - unused_pooler
    if missing_backbone:
        raise RuntimeError(f"Missing backbone parameters: {sorted(missing_backbone)}")
    for field in ("mismatched_keys", "error_msgs"):
        if loading_info.get(field):
            raise RuntimeError(f"RNA-FM loading failed: {field}={loading_info[field]}")
    # Only pretraining/secondary-structure heads may be unused by the backbone.
    unexpected = loading_info.get("unexpected_keys", [])
    if any(not key.startswith(("lm_head.", "ss_head.")) for key in unexpected):
        raise RuntimeError(f"Unexpected backbone parameters: {unexpected}")
    if model.config.hidden_size != Config.EMBEDDING_DIM:
        raise ValueError(f"Expected hidden_size=640, got {model.config.hidden_size}")

    inputs = tokenizer(
        ["ACGU", "ACGUACGU"], padding=True, truncation=True,
        max_length=Config.MODEL_MAX_LENGTH, return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        hidden = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
    expected = (*inputs["input_ids"].shape, Config.EMBEDDING_DIM)
    if tuple(hidden.shape) != expected or not torch.isfinite(hidden).all():
        raise RuntimeError(f"Invalid last_hidden_state: {tuple(hidden.shape)}")
    print(f"Load and forward check passed: {tuple(hidden.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(Config.PRETRAINED_MODEL_NAME))
    args = parser.parse_args()
    check_model(args.model_dir)

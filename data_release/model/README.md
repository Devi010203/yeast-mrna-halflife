## Installing the pretrained RNA-FM backbone

The entry point is [`code_release/src/main.py`](../../code_release/src/main.py).
It imports `RnaFmModel` and `RnaTokenizer` from **MultiMolecule**, with
`multimolecule==0.0.8`, `transformers==4.54.0`, and `torch==2.7.1` pinned in the
project environment. It loads a local model/tokenizer directory through
`from_pretrained()`, then sends `last_hidden_state` to a 640-dimensional
Transformer regression head. `MODEL_MAX_LENGTH=448` includes special tokens.

The original [`RNA-FM_pretrained.pth`](https://huggingface.co/cuhkaih/rnafm/tree/main)
is not directly compatible with these calls. Do not rename it to
`pytorch_model.bin`, create `config.json` by hand, or mix tokenizers from other
models. Convert the checkpoint using the converter in the pinned environment.

### 1. Install the project environment

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
git clone https://github.com/Devi010203/yeast-mrna-halflife.git
cd yeast-mrna-halflife/code_release/src
uv sync --locked
```

The dependency files are in `code_release/src`, not the repository root.
Run all commands below from this directory. Python 3.12 is suitable for the
locked environment. GPU availability affects training speed, but the loading
check below runs on CPU.

### 2. Download the original checkpoint

```sh
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cuhkaih/rnafm', revision='91d4a46d28d8054a7b429955e8fc0c253ba0afd6', filename='RNA-FM_pretrained.pth', local_dir='.')"
```

This pins the public source snapshot. Use **RNA-FM_pretrained.pth**;
`mRNA-FM_pretrained.pth` is a different model (1280-dimensional, codon-based),
and the files under `SS/` are downstream secondary-structure checkpoints.
Neither is a drop-in replacement for this project's 640-dimensional backbone.

### 3. Convert into a new directory

```sh
uv run python -m multimolecule.models.rnafm.convert_checkpoint --checkpoint_path RNA-FM_pretrained.pth --output_path rnafm-converted
```

**Pass the relative checkpoint filename exactly as shown.** The 0.0.8 converter
selects its mRNA/CDS branch when the supplied checkpoint path contains `mrna`
or `cds`. An absolute path containing this repository's name,
`yeast-mrna-halflife`, therefore selects the wrong branch.

Use a **new** output directory: the upstream converter deletes an existing
output directory before saving. It deserializes the checkpoint with
`weights_only=False`; only use a checkpoint from a source you trust.

The converter maps parameter names and vocabulary-dependent embeddings, loads
the converted state dictionary strictly, and saves configuration, weights and
tokenizer files together. Investigate any conversion error; do not bypass it
with `strict=False`.

After conversion succeeds, install the generated directory:

```sh
uv run python -c "from shutil import copytree; copytree('rnafm-converted', '../../data_release/model/rna-fm')"
```

The copy command requires the destination not to exist. Keep an existing model
bundle separately before installing a replacement.

```text
data_release/model/rna-fm/
├── config.json
├── model.safetensors
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
└── other files saved by the converter
```

Either saved weight format can be loaded; keep the matching configuration and
tokenizer. `tokenizer.json` is not required for this tokenizer.

### 4. Check loading, then train

```sh
uv run check_rnafm.py
uv run main.py
```

The first command loads only local files, checks for missing/mismatched backbone
parameters and unexpected non-head parameters, verifies `hidden_size=640`, and
runs a padded batch of two different-length sequences. It must print
`Load and forward check passed` before training starts.

The original checkpoint has no `pooler.dense.weight` or `pooler.dense.bias`.
MultiMolecule initializes those two optional pooling parameters when loading
`RnaFmModel`; they do not affect `last_hidden_state`, the output used by this
project. The check permits only those two missing parameters and unused
`lm_head.*` / `ss_head.*` pretraining-head parameters. Missing embedding or
encoder parameters, other unexpected parameters, and shape mismatches fail.

`Config.PRETRAINED_MODEL_NAME` resolves to the repository's
`data_release/model/rna-fm` directory independently of the working directory.
No configuration edit is needed for the installation above. To check a different
converted bundle, use `uv run check_rnafm.py --model-dir /path/to/bundle` and
set `Config.PRETRAINED_MODEL_NAME` to the same directory before training.

### Other loading options

- [`multimolecule/rnafm`](https://huggingface.co/multimolecule/rnafm) is an
  **unofficial MultiMolecule implementation/conversion**, with the expected
  configuration, weights and tokenizer files. Its current `main` is not pinned
  to this project's 0.0.8 dependency. Before recommending a preconverted snapshot,
  verify it with `check_rnafm.py` in the locked environment and record its full
  revision. Do not assume that the latest snapshot is compatible.
- The original `.pth` can be loaded through
  `fm.pretrained.rna_fm_t12(model_location=...)` in the original RNA-FM package.
  Integrating that API here requires changes to batching/tokenization, token IDs,
  padding masks and the forward-output adapter; replacing only the loading call
  is insufficient. The current training code uses the converted workflow above.

This installs the **pretrained backbone**, not a trained yeast half-life
regressor. Successful installation does not by itself establish reproduction of
the paper's metrics or identify the exact checkpoint used in the original runs.

### Validation record

On 2026-10-01, the pinned original checkpoint above was downloaded and converted
in the locked environment on Windows with Python 3.12.0. Strict conversion and
the CPU loading/forward check passed with output shape `(2, 10, 640)`.
No full cross-validation or paper-metric reproduction was run.

On this Windows/Python combination, the pinned transitive `multiprocess`
dependency also emitted an `Exception ignored in ResourceTracker.__del__`
message about `RLock._recursion_count` during process shutdown. This was
observed after successful conversion/loading and exit code 0; the generated
bundle and forward result were checked independently.

### Source references

- [Installation issue #1](https://github.com/Devi010203/yeast-mrna-halflife/issues/1)
- [Original RNA-FM loading code](https://github.com/ml4bio/RNA-FM/blob/main/fm/pretrained.py)
- [MultiMolecule 0.0.8 conversion code](https://github.com/DLS5-Omics/multimolecule/blob/v0.0.8/multimolecule/models/rnafm/convert_checkpoint.py)
- [MultiMolecule 0.0.8 save logic](https://github.com/DLS5-Omics/multimolecule/blob/v0.0.8/multimolecule/models/conversion_utils.py)

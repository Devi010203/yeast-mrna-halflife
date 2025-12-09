import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False.*")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from multimolecule import RnaFmModel, RnaTokenizer
from tqdm import tqdm
import os
from datetime import datetime
from functools import partial
import time
import json
from scipy.stats import pearsonr, spearmanr
import platform
import math
from pathlib import Path



class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT.parent / "data_release"
    DATA_PATH = DATA_ROOT / "processed" / "mRNA_half_life_dataset_RNA.csv"
    SEQ_COLUMN = 'sequence'
    TARGET_COLUMN = 'Isoform Half-Life'
    LOG1P_TARGET = True

    LEARNING_RATE = 2e-5
    DROPOUT = 0.1

    MODEL_MAX_LENGTH = 1026
    EMBEDDING_DIM = 640
    PRETRAINED_MODEL_NAME = DATA_ROOT / "model" / "rna-fm"

    TEST_SET_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    EPOCHS = 20
    RANDOM_SEED = 42

    EARLY_STOPPING_PATIENCE = 5

    LOG_BASE_PATH = 'log'
    MODEL_SAVE_NAME = 'mrna_transformer_accumulation_model.pth'

    TOKEN_ATTN_HEADS = 10
    TOKEN_ATTN_LAYERS = 3
    TOKEN_ATTN_FF_MULT = 4
    TOKEN_HEAD_MLP_HIDDEN = 512

    WEIGHT_DECAY = 1e-2
    EMA_DECAY = 0.999


# --- 2) Utils ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def create_experiment_directory(config, script_file_path):
    script_dir = os.path.dirname(os.path.abspath(script_file_path))
    project_root = os.path.dirname(script_dir)
    log_base_dir = os.path.join(project_root, config.LOG_BASE_PATH)

    base_name = f"{os.path.splitext(os.path.basename(script_file_path))[0]}_{datetime.now().strftime('%Y%m%d')}"
    run_index = 1
    while True:
        experiment_dir_name = f"{base_name}_{run_index:02d}"
        experiment_path = os.path.join(log_base_dir, experiment_dir_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
            break
        run_index += 1
    tensorboard_log_path = os.path.join(experiment_path, 'tensorboard-log')
    os.makedirs(tensorboard_log_path, exist_ok=True)
    return experiment_path, tensorboard_log_path


# --- 3) Data ---
class MRNADataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': str(self.sequences[idx]),
            'target': torch.tensor(self.targets[idx], dtype=torch.float)
        }

def collate_fn_no_chunk(batch, tokenizer, config):
    sequences = [item['sequence'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    tokenized = tokenizer(
        sequences, padding=True, truncation=True, max_length=config.MODEL_MAX_LENGTH, return_tensors="pt"
    )
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'targets': targets,
        'sequence': sequences
    }


# --- 4) Model ---
class _SinPosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)

class _ResidualMLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act(self.fc1(self.pre(x))))
        h = self.drop2(self.fc2(h))
        x = x + h
        return self.out(x).squeeze(-1)

class TokenTransformerHead(nn.Module):
    def __init__(self, dim: int, nhead: int = 10, num_layers: int = 3,
                 ff_mult: int = 4, dropout: float = 0.1, mlp_hidden: int = 512):
        super().__init__()
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ff_mult * dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = _SinPosEnc(dim)
        self.regressor = _ResidualMLPHead(dim, mlp_hidden, dropout)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = token_embeds.shape
        x = torch.cat([self.cls.expand(B, 1, D), token_embeds], dim=1)  # [B, L+1, D]
        if attention_mask is not None:
            kpm = torch.ones(B, L + 1, dtype=torch.bool, device=token_embeds.device)
            kpm[:, 0] = False
            kpm[:, 1:] = (attention_mask == 0)
        else:
            kpm = None
        x = self.posenc(x)
        x = self.encoder(x, src_key_padding_mask=kpm)
        cls = x[:, 0, :]
        return self.regressor(cls)

class ChunkingMRNATransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(f"The pre-trained RNA-FM model is currently being loaded from a local path.: {config.PRETRAINED_MODEL_NAME}")
        self.bert = RnaFmModel.from_pretrained(config.PRETRAINED_MODEL_NAME, trust_remote_code=True)
        self.token_head = TokenTransformerHead(
            dim=config.EMBEDDING_DIM,
            nhead=getattr(config, "TOKEN_ATTN_HEADS", 10),
            num_layers=getattr(config, "TOKEN_ATTN_LAYERS", 3),
            ff_mult=getattr(config, "TOKEN_ATTN_FF_MULT", 4),
            dropout=getattr(config, "DROPOUT", 0.1),
            mlp_hidden=getattr(config, "TOKEN_HEAD_MLP_HIDDEN", 512),
        )
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        return self.token_head(token_embeddings, attention_mask)


# --- 4.5) EMA ---
class ModelEMA:
    """Call `update()` after `optimizer.step()`; During evaluation, switch/restore weights using `apply_to()`/`restore()`."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.backup = None
    @torch.no_grad()
    def update(self, model: nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            i += 1
    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = []
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            self.backup.append(p.detach().clone())
            p.data.copy_(self.shadow[i])
            i += 1
    @torch.no_grad()
    def restore(self, model: nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[i])
            i += 1
        self.backup = None


# --- 5) Training / Eval ---
class EarlyStopper:
    """Universal Early Stopping: Monitoring the 'bigger is better' metric (using Spearman's coefficient here)."""
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = -np.inf
        self.early_stop = False
    def __call__(self, metric_value):
        if metric_value > self.best_metric + self.min_delta:
            self.best_metric = metric_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def train_epoch(model, data_loader, loss_fn, optimizer, device, scaler, config, scheduler=None, ema: 'ModelEMA' = None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    accum = config.GRAD_ACCUMULATION_STEPS
    for i, batch in enumerate(tqdm(data_loader, desc="During training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(input_ids, attention_mask)          # [B]
            loss = loss_fn(outputs, targets) / accum


        scaler.scale(loss).backward()
        total_loss += loss.detach() * accum

        ready_to_step = ((i + 1) % accum == 0)
        if ready_to_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

    # Processing the final small segment of residue (when the sample count is not an integer multiple of accum)
    # Note: This is only required if the loop has run at least once and the final alignment does not match accum
    num_batches = len(data_loader)
    if num_batches > 0 and (num_batches % accum != 0):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

    return (total_loss / len(data_loader)).item()

def evaluate(model, data_loader, loss_fn, device, config, ema: 'ModelEMA' = None):
    if ema is not None:
        ema.apply_to(model)
    model.eval()
    total_loss, all_preds, all_targets, all_sequences = 0, [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Under assessment", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            sequences = batch.get('sequence') if 'sequence' in batch else None
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            if sequences is not None:
                all_sequences.extend(sequences)
    avg_loss = total_loss / len(data_loader)


    all_targets = np.array(all_targets).reshape(-1)
    all_preds = np.array(all_preds).reshape(-1)

    # Determine whether to perform expm1 restoration based on config.LOG1P_TARGET.
    if config.LOG1P_TARGET:
        true_targets = np.expm1(all_targets)
        pred_targets = np.expm1(all_preds)
    else:
        true_targets = all_targets
        pred_targets = all_preds

    r2 = r2_score(true_targets, pred_targets)
    mse = mean_squared_error(true_targets, pred_targets)
    pearson_corr, _ = pearsonr(true_targets, pred_targets)
    spearman_corr, _ = spearmanr(true_targets, pred_targets)

    if ema is not None:
        ema.restore(model)
    return avg_loss, r2, mse, pearson_corr, spearman_corr, true_targets, pred_targets, all_sequences

def make_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """ per-step cosine scheduler with linear warmup """
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- 6) Main ---
if __name__ == '__main__':
    config = Config()
    set_seed(config.RANDOM_SEED)
    device = get_device()
    output_dir, tensorboard_log_dir = create_experiment_directory(config, __file__)

    print("--- Configuration Information ---")
    print(f"Equipment: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Batch Size: {config.BATCH_SIZE}, Grad Accumulation Steps: {config.GRAD_ACCUMULATION_STEPS}, "
          f"Effective Batch Size: {config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS}")
    print("---------------------")

    env_info = {
        "random_seed": config.RANDOM_SEED,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python_version": platform.python_version()
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=4)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_file_path = os.path.join(project_root, os.path.normpath(config.DATA_PATH))
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    df = pd.read_csv(data_file_path)

    # 1) Check whether the column exists
    req_cols = [config.SEQ_COLUMN, config.TARGET_COLUMN]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"The following data is missing from the dataset:: {missing}，Please check the settings in the column name or Config.。")

    # 2) Discard missing values
    df.dropna(subset=req_cols, inplace=True)

    # 3) Uniformly mapped to internal column names：sequence / target
    df['sequence'] = df[config.SEQ_COLUMN].astype(str)

    if config.LOG1P_TARGET:
        # Training in log1p space
        df['target'] = np.log1p(df[config.TARGET_COLUMN].astype(float))
    else:
        df['target'] = df[config.TARGET_COLUMN].astype(float)

    # df = pd.read_csv(data_file_path)
    # df.dropna(subset=['sequence', 'Isoform Half-Life'], inplace=True)
    # df['target'] = np.log1p(df['Isoform Half-Life'])

    # Training/validation/test split
    df_train_val, df_test = train_test_split(df, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_SEED)
    print(f"Data segmentation -> Training and validation sets: {len(df_train_val)}, test set: {len(df_test)}")

    # tokenizer & collate

    tokenizer = RnaTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME, trust_remote_code=True)
    collate_with_chunking = partial(collate_fn_no_chunk, tokenizer=tokenizer, config=config)

    # --- K fold ---
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    val_spearman_scores = []
    results_log = []
    fold_best_metrics = []

    print("--- Commencing K-fold inspection ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train_val), start=1):
        print(f"\n=== Fold {fold}/5 ===")

        df_train = df_train_val.iloc[train_idx]
        df_val = df_train_val.iloc[val_idx]

        train_dataset = MRNADataset(df_train['sequence'].values, df_train['target'].values)
        val_dataset = MRNADataset(df_val['sequence'].values, df_val['target'].values)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_with_chunking)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                collate_fn=collate_with_chunking)
        print("The data loader has been created (dynamic filling).")

        writer = SummaryWriter(log_dir=os.path.join(tensorboard_log_dir, f"fold_{fold}"))

        model = ChunkingMRNATransformer(config).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

        # per-step cosine + 5% warmup
        steps_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUMULATION_STEPS)
        num_training_steps = steps_per_epoch * config.EPOCHS
        num_warmup_steps = max(1, int(0.05 * num_training_steps))
        scheduler = make_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        ema = ModelEMA(model, decay=getattr(config, "EMA_DECAY", 0.999))
        early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, min_delta=1e-4)

        best_val_s = -np.inf
        best_model_state = None
        lr_schedule = []
        start_time = time.time()

        for epoch in range(config.EPOCHS):
            print(f"Epoch {epoch + 1}/{config.EPOCHS}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, config,
                                     scheduler=scheduler, ema=ema)
            val_loss, val_r2, val_mse, val_pearson, val_spearman, y_true, y_pred, _ = evaluate(
                model, val_loader, loss_fn, device, config, ema=ema
            )

            print(f"Epoch {epoch + 1} | Training loss: {train_loss:.4f} | Verification loss: {val_loss:.4f} "
                  f"| Verification R²: {val_r2:.4f} | Pearson: {val_pearson:.4f} | Spearman: {val_spearman:.4f}")

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
            writer.add_scalar('R2/val', val_r2, epoch + 1)
            writer.add_scalar('MSE/val', val_mse, epoch + 1)
            writer.add_scalar('Pearson/val', val_pearson, epoch + 1)
            writer.add_scalar('Spearman/val', val_spearman, epoch + 1)
            writer.add_scalar('LearningRate', current_lr, epoch + 1)
            lr_schedule.append((epoch + 1, current_lr))

            # Using Spearman's correlation coefficient as the basis for 'best/early termination'
            if val_spearman > best_val_s:
                best_val_s = val_spearman
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, os.path.join(output_dir, f"best_model_fold{fold}.pth"))

            _ = early_stopper(val_spearman)
            if early_stopper.early_stop:
                print(f"Early stopping was triggered at epoch {epoch + 1} (based on Spearman).")
                break

            results_log.append({
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_r2": float(val_r2),
                "val_mse": float(val_mse),
                "val_pearson": float(val_pearson),
                "val_spearman": float(val_spearman),
                "lr": float(current_lr)
            })

        runtime_log = {
            "fold": fold,
            "runtime_sec": time.time() - start_time,
            "early_stopped_epoch": (epoch + 1) if early_stopper.early_stop else None
        }
        with open(os.path.join(output_dir, f"runtime_log_fold{fold}.json"), "w") as f:
            json.dump(runtime_log, f, indent=4)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Fold {fold}: The optimal model for the validation set has been loaded. (Spearman={best_val_s:.4f})。")

        writer.close()
        val_spearman_scores.append(best_val_s)
        fold_best_metrics.append({"fold": fold, "best_val_spearman": best_val_s})
        print(f"Fold {fold} Optimal Verification Spearman = {best_val_s:.4f}")

        with open(os.path.join(output_dir, f"learning_rate_schedule_fold{fold}.csv"), "w") as f:
            f.write("epoch,lr\n")
            for e, lr in lr_schedule:
                f.write(f"{e},{lr}\n")

        # Save the last validated prediction for this fold (if available)
        try:
            df_val_pred = pd.DataFrame({"true": y_true, "pred": y_pred})
            df_val_pred.to_csv(os.path.join(output_dir, f"val_predictions_fold{fold}.csv"), index=False)
        except Exception:
            pass

    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(results_log, f, indent=4)
    with open(os.path.join(output_dir, "cv_fold_best_metrics.json"), "w") as f:
        json.dump(fold_best_metrics, f, indent=4)

    cv_summary = pd.DataFrame({
        "fold": list(range(1, len(val_spearman_scores) + 1)),
        "val_spearman": val_spearman_scores
    })
    cv_summary["mean_spearman"] = np.mean(val_spearman_scores)
    cv_summary.to_csv(os.path.join(output_dir, "cv_summary.csv"), index=False)

    print("\n=== K-fold cross-validation completed ===")
    print(f"Verification at each fold Spearman: {val_spearman_scores}")
    print(f"Average verification Spearman: {np.mean(val_spearman_scores):.4f}")

    # --- Final Train + Test ---
    print("\n=== Retrain using the entire training dataset (train_val) and evaluate on the test set. ===")
    df_train, df_val = train_test_split(df_train_val, test_size=config.VALIDATION_SPLIT,
                                        random_state=config.RANDOM_SEED)

    train_dataset = MRNADataset(df_train['sequence'].values, df_train['target'].values)
    val_dataset = MRNADataset(df_val['sequence'].values, df_val['target'].values)
    test_dataset = MRNADataset(df_test['sequence'].values, df_test['target'].values)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_chunking)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_with_chunking)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_with_chunking)

    writer = SummaryWriter(log_dir=os.path.join(tensorboard_log_dir, "final_run"))

    model = ChunkingMRNATransformer(config).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUMULATION_STEPS)
    num_training_steps = steps_per_epoch * config.EPOCHS
    num_warmup_steps = max(1, int(0.05 * num_training_steps))
    scheduler = make_cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=getattr(config, "EMA_DECAY", 0.999))
    early_stopper = EarlyStopper(patience=config.EARLY_STOPPING_PATIENCE, min_delta=1e-4)

    best_val_s = -np.inf
    best_model_state = None
    results_log_final = []
    lr_schedule_final = []
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, config,
                                 scheduler=scheduler, ema=ema)
        val_loss, val_r2, val_mse, val_pearson, val_spearman, y_true_val, y_pred_val, _ = evaluate(
            model, val_loader, loss_fn, device, config, ema=ema
        )

        print(f"Epoch {epoch + 1} | Training loss: {train_loss:.4f} | Verification loss: {val_loss:.4f} "
              f"| Verification R²: {val_r2:.4f} | Pearson: {val_pearson:.4f} | Spearman: {val_spearman:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('R2/val', val_r2, epoch + 1)
        writer.add_scalar('MSE/val', val_mse, epoch + 1)
        writer.add_scalar('Pearson/val', val_pearson, epoch + 1)
        writer.add_scalar('Spearman/val', val_spearman, epoch + 1)
        writer.add_scalar('LearningRate', current_lr, epoch + 1)
        lr_schedule_final.append((epoch + 1, current_lr))

        # spearman as the optimal criterion
        if val_spearman > best_val_s:
            best_val_s = val_spearman
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(output_dir, "best_model_final.pth"))

        _ = early_stopper(val_spearman)
        if early_stopper.early_stop:
            print(f"Early stopping was triggered at epoch {epoch + 1} (based on Spearman).")
            break

        results_log_final.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_r2": float(val_r2),
            "val_mse": float(val_mse),
            "val_pearson": float(val_pearson),
            "val_spearman": float(val_spearman),
            "lr": float(current_lr)
        })

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"The optimal model from the final training stage validation set (Spearman={best_val_s:.4f}) has been loaded for evaluation on the test set.")

    runtime_log_final = {
        "runtime_sec": time.time() - start_time,
        "early_stopped_epoch": (epoch + 1) if early_stopper.early_stop else None
    }
    with open(os.path.join(output_dir, "runtime_log_final.json"), "w") as f:
        json.dump(runtime_log_final, f, indent=4)

    pd.DataFrame(results_log_final).to_csv(os.path.join(output_dir, "training_curve_final.csv"), index=False)

    # Test set evaluation
    test_loss, test_r2, test_mse, test_pearson, test_spearman, y_true_test, y_pred_test, seqs_test = evaluate(
        model, test_loader, loss_fn, device, config, ema=ema
    )

    with open(os.path.join(output_dir, "learning_rate_schedule_final.csv"), "w") as f:
        f.write("epoch,lr\n")
        for e, lr in lr_schedule_final:
            f.write(f"{e},{lr}\n")

    try:
        df_val_pred_final = pd.DataFrame({"true": y_true_val, "pred": y_pred_val})
        df_val_pred_final.to_csv(os.path.join(output_dir, "val_predictions_final.csv"), index=False)
    except Exception:
        pass

    with open(os.path.join(output_dir, "final_training_log.json"), "w") as f:
        json.dump(results_log_final, f, indent=4)

    print(" == = Final test set performance == = ")
    print(f"Test Loss: {test_loss:.4f} | Test R²: {test_r2:.4f} | Test MSE: {test_mse:.4f} "
          f"| Pearson: {test_pearson:.4f} | Spearman: {test_spearman:.4f}")

    if seqs_test:
        df_results = pd.DataFrame({
            "sequence": seqs_test,
            "true": y_true_test,
            "pred": y_pred_test
        })
    else:
        df_results = df_test.copy()
        if config.LOG1P_TARGET:
            df_results["true"] = np.expm1(df_test["target"].values)
        else:
            df_results["true"] = df_test["target"].values
        df_results["pred"] = y_pred_test

    df_results.to_csv(os.path.join(output_dir, "final_test_predictions.csv"), index=False)

    final_metrics = {
        "test_loss": float(test_loss),
        "test_r2": float(test_r2),
        "test_mse": float(test_mse),
        "test_pearson": float(test_pearson),
        "test_spearman": float(test_spearman)
    }
    with open(os.path.join(output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Additional statistics
    try:
        df_results["residual"] = df_results["true"] - df_results["pred"]
        df_results["abs_error"] = np.abs(df_results["residual"])
        df_results[["residual", "abs_error"]].describe().to_csv(os.path.join(output_dir, "residual_summary.csv"))
    except Exception:
        pass

    # Record test set metrics
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_log_dir, "final_run"))
    writer.add_scalar('Loss/test', test_loss, (epoch + 1))
    writer.add_scalar('R2/test', test_r2, (epoch + 1))
    writer.add_scalar('MSE/test', test_mse, (epoch + 1))
    writer.add_scalar('Pearson/test', test_pearson, (epoch + 1))
    writer.add_scalar('Spearman/test', test_spearman, (epoch + 1))
    try:
        writer.add_histogram("Residuals/test", (df_results["true"] - df_results["pred"]).values, (epoch + 1))
    except Exception:
        pass
    writer.close()

    # # Save model
    # torch.save(model.state_dict(), os.path.join(output_dir, config.MODEL_SAVE_NAME))
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # torch.save(model.state_dict(), os.path.join(output_dir, f"model_final_{timestamp}.pth"))

    # Save config
    try:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=4)
    except Exception:
        pass

    print(f"Saved and exported to: {output_dir}")

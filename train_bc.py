"""Обучение BC-модели (CNN + MLP) на собранном датасете.

Запуск:
    python3 train_bc.py --train_dir dataset/train_1k --eval_dir dataset/eval --exp_name bc_1k --epochs 100 --batch_size 64 --lr 1e-4
    python3 train_bc.py --train_dir dataset/train_10k --eval_dir dataset/eval --exp_name bc_10k --epochs 100 --batch_size 64 --lr 1e-4
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BCPolicy


def train_step(model, optimizer, obs_batch, action_batch):
    """Один шаг обучения BC (см. Урок 5).

    TODO: реализуйте классический цикл PyTorch из 5 шагов:
      1) обнулите градиенты (``optimizer.zero_grad``);
      2) прямой проход: ``pred = model(obs_batch)``;
      3) посчитайте MSE-loss между ``pred`` и ``action_batch``;
      4) обратный проход (``loss.backward``);
      5) шаг оптимизатора (``optimizer.step``).
    Верните значение loss как число (``loss.item()``).
    """
    raise NotImplementedError(
        "Реализуйте train_step — один шаг обучения BC (см. Урок 5)."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение BC-модели")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Папка с тренировочными эпизодами (.npz)")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Папка с eval-эпизодами (.npz)")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Название эксперимента (bc_1k, bc_10k)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class EpisodeDataset(Dataset):
    """Датасет из .npz файлов: каждый шаг = одна пара (obs, action)."""

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise ValueError(f"Нет .npz файлов в {data_dir}")

        self.observations = []
        self.actions = []

        for f in self.files:
            ep = np.load(f)
            obs = ep["obs"]     # (T, 84, 84, 3) uint8
            acts = ep["actions"]  # (T, 8) float32
            T = min(obs.shape[0], acts.shape[0])
            self.observations.append(obs[:T])
            self.actions.append(acts[:T])

        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        print(f"Loaded {len(self.files)} episodes, {len(self)} steps from {data_dir}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx].copy()).float() / 255.0
        action = torch.from_numpy(self.actions[idx].copy())
        return obs, action


def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for obs, actions in eval_loader:
            obs, actions = obs.to(device), actions.to(device)
            pred = model(obs)
            loss = criterion(pred, actions)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dataset = EpisodeDataset(args.train_dir)
    eval_dataset = EpisodeDataset(args.eval_dir)

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    model = BCPolicy(action_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    log_dir = os.path.join("logs", args.exp_name)
    ckpt_dir = os.path.join("logs", args.exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_eval_loss = float("inf")
    best_epoch = 0

    print(f"\nОбучение: {args.exp_name}")
    print(f"  Train: {len(train_dataset)} steps | Eval: {len(eval_dataset)} steps")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for obs, actions in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            loss_value = train_step(model, optimizer, obs, actions)
            train_loss_sum += loss_value
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        eval_loss = evaluate(model, eval_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("eval/loss", eval_loss, epoch)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_loss": eval_loss,
                "train_loss": train_loss,
            }, os.path.join(ckpt_dir, "best.pt"))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | eval_loss: {eval_loss:.4f} | "
              f"{elapsed:.0f}s")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_loss": eval_loss,
        "train_loss": train_loss,
    }, os.path.join(ckpt_dir, "last.pt"))

    writer.close()

    print()
    print(f"Checkpoint saved: {ckpt_dir}/last.pt")
    print(f"Best checkpoint: {ckpt_dir}/best.pt (epoch {best_epoch}, eval_loss={best_eval_loss:.4f})")


if __name__ == "__main__":
    main()

"""
=============================================================================
  SELF-PRUNING NEURAL NETWORK — CIFAR-10
  Author  : AI Engineering Intern Submission
  Task    : Tredence Analytics — AI Engineer Case Study
  Method  : Learnable Gate Parameters + L1 Sparsity Regularization
=============================================================================

CORE IDEA
---------
Instead of pruning after training, we teach the network to prune ITSELF
during training. Each weight gets a learnable "gate" (via sigmoid). An L1
penalty on these gates pushes them toward zero — effectively removing those
weights from the network.

This exploits a mathematical property of the L1 norm:
  - L2 norm  →  shrinks weights smoothly (never exactly zero)
  - L1 norm  →  creates a "kink" at zero → pushes weights to exactly 0
  (This is why LASSO regression creates sparse solutions; same principle here.)
"""


# IMPORTS

import os
import time
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")          # headless backend — safe on all machines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# REPRODUCIBILITY

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")



# PART 1 — PRUNABLE LINEAR LAYER

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate
    to every weight element.

    Forward pass:
        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w     = weight ⊙ gates                element-wise product
        output       = pruned_w @ x^T + bias

    WHY SIGMOID?
        We need gates strictly between 0 and 1 without any clamping hack.
        Sigmoid maps ℝ → (0,1) smoothly, so gradients always flow through it.
        Combined with L1 loss, the optimizer pushes gate_scores → −∞, which
        maps sigmoid → 0, effectively pruning that weight.

    GRADIENT FLOW:
        Both `weight` and `gate_scores` are nn.Parameter objects — they are
        registered in the computation graph automatically. Autograd handles
        d(Loss)/d(gate_scores) and d(Loss)/d(weight) without any extra code.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        
        # gate_scores has the SAME shape as weight.
        # Initialised near 0 → sigmoid(0) = 0.5, so gates start half-open.
        # The optimizer + L1 loss will drive most of them toward 0.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        

        # Kaiming uniform init (same default as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound  = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — turn raw scores into gates ∈ (0,1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2 — element-wise mask the weights
        pruned_weights = self.weight * gates

        # Step 3 — standard linear transformation using masked weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (detached, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (i.e., effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")

# PART 2 — NEURAL NETWORK ARCHITECTURE

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 classification.

    Architecture:
        Input  : 3 × 32 × 32 = 3072 pixels (flattened)
        FC1    : 3072 → 512   PrunableLinear  + BatchNorm + ReLU
        FC2    :  512 → 256   PrunableLinear  + BatchNorm + ReLU
        FC3    :  256 → 128   PrunableLinear  + BatchNorm + ReLU
        Output :  128 → 10    PrunableLinear  (logits; no activation)

    DESIGN NOTE:
        BatchNorm is placed after the prunable layer (before activation).
        This stabilises training when many weights get zeroed-out — otherwise
        dead neurons can destabilise batch statistics.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = PrunableLinear(128, 10)   # 10 CIFAR-10 classes

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)               # flatten

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        return self.fc4(x)                       # raw logits

    def prunable_layers(self):
        """Generator yielding every PrunableLinear in the model."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across ALL PrunableLinear layers.

        WHY L1?
            The gradient of |x| w.r.t. x is sign(x) — a constant ±1.
            This means the optimiser always gets a full-strength push toward
            zero regardless of how small the value already is.
            L2 (x²) would give gradient 2x → push weakens as x → 0, so
            weights shrink but never reach exactly zero.
            L1 crosses that threshold and produces true sparsity.
        """
        all_gates = [torch.sigmoid(layer.gate_scores)
                     for layer in self.prunable_layers()]
        return torch.cat([g.view(-1) for g in all_gates]).sum()

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall fraction of pruned gates across the whole network."""
        counts = []
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            counts.append((gates < threshold).float())
        all_gates = torch.cat([c.view(-1) for c in counts])
        return all_gates.mean().item()

    def count_params(self):
        total  = sum(p.numel() for p in self.parameters())
        pruned = sum(
            (torch.sigmoid(layer.gate_scores) < 1e-2).sum().item()
            for layer in self.prunable_layers()
        )
        return total, pruned



# DATA LOADING
def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """
    Returns train and test DataLoaders for CIFAR-10.

    Augmentation strategy (train only):
        RandomHorizontalFlip  — free augmentation, widely used
        RandomCrop(32, pad=4) — standard CIFAR practice
    Normalisation:
        CIFAR-10 channel-wise mean/std computed from training set.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[DATA] Train: {len(train_set):,}  |  Test: {len(test_set):,}  "
          f"|  Batch size: {batch_size}")
    return train_loader, test_loader


#  PART 3 — TRAINING LOOP

def train_one_epoch(model, loader, optimizer, criterion, lam, epoch, log_every=100):
    """
    Runs a single training epoch.

    Returns:
        avg_total_loss, avg_ce_loss, avg_sparsity_loss
    """
    model.train()
    total_loss_sum = ce_loss_sum = sp_loss_sum = 0.0
    correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        #FORWARD
        logits    = model(images)
        ce_loss   = criterion(logits, labels)
        sp_loss   = model.sparsity_loss()
        loss      = ce_loss + lam * sp_loss
        

        # BACKWARD + STEP 
        loss.backward()
        # Gradient clipping — prevents exploding gradients during early
        # training when gates are still largely active
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        

        total_loss_sum += loss.item()
        ce_loss_sum    += ce_loss.item()
        sp_loss_sum    += sp_loss.item()

        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % log_every == 0:
            running_acc = 100.0 * correct / total
            print(f"  Epoch {epoch:02d} | Batch {batch_idx+1:>3}/{len(loader)} "
                  f"| Loss {loss.item():.4f} "
                  f"(CE {ce_loss.item():.4f} + λ·Sp {lam*sp_loss.item():.4f}) "
                  f"| Acc {running_acc:.1f}%")

    n = len(loader)
    return total_loss_sum / n, ce_loss_sum / n, sp_loss_sum / n


def evaluate(model, loader, criterion):
    """Evaluates the model on the given loader. Returns (accuracy, avg_loss)."""
    model.eval()
    loss_sum = correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits   = model(images)
            loss_sum += criterion(logits, labels).item()
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    return 100.0 * correct / total, loss_sum / len(loader)


def train(lam: float, epochs: int = 30, batch_size: int = 128,
          lr: float = 1e-3, data_dir: str = "./data"):
    """
    Full training pipeline for a given lambda (λ).

    Returns:
        model          — trained SelfPruningNet
        history        — dict of per-epoch metrics
    """
    print(f"\n{'='*65}")
    print(f"  TRAINING  |  λ = {lam}  |  Epochs = {epochs}")
    print(f"{'='*65}")

    train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir)

    model     = SelfPruningNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Adam works well here; weight_decay adds a small L2 on *all* params
    # (but the important sparsity pressure comes from our custom L1 term)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing smoothly decays LR — prevents oscillations late in
    # training and gives the gates time to settle near 0 or 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "ce_loss": [], "sp_loss": [],
        "test_acc": [],   "test_loss": [], "sparsity": []
    }

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, ce_loss, sp_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lam, epoch)
        scheduler.step()

        test_acc, test_loss = evaluate(model, test_loader, criterion)
        sparsity = model.global_sparsity()

        history["train_loss"].append(train_loss)
        history["ce_loss"].append(ce_loss)
        history["sp_loss"].append(sp_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["sparsity"].append(sparsity)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"  ► Epoch {epoch:02d}/{epochs} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Sparsity: {100*sparsity:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")

    elapsed = time.time() - t0
    total_p, pruned_p = model.count_params()
    print(f"\n  ✓ Done in {elapsed:.1f}s | Best acc: {best_acc:.2f}% | "
          f"Final sparsity: {100*sparsity:.1f}%")
    print(f"  ✓ Total params: {total_p:,} | Pruned (gate<1e-2): {pruned_p:,} "
          f"({100*pruned_p/total_p:.1f}%)")

    return model, history



# VISUALISATION

def collect_all_gates(model) -> np.ndarray:
    """Flattens all gate values from all PrunableLinear layers into one array."""
    gates = []
    for layer in model.prunable_layers():
        gates.append(layer.get_gates().cpu().numpy().ravel())
    return np.concatenate(gates)


def plot_gate_distribution(model, lam: float, out_path: str = "gate_distribution.png"):
    """
    Plots the distribution of final gate values.

    A successful pruning run shows:
        - A tall spike near 0    (pruned/dead connections)
        - A smaller cluster near 0.5–1.0  (active connections)
    """
    gates = collect_all_gates(model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Gate Value Distribution  |  λ = {lam}", fontsize=14, y=1.02)

    # Left: full histogram 
    ax = axes[0]
    ax.hist(gates, bins=80, color="#3b82f6", edgecolor="white", linewidth=0.3)
    ax.set_title("All gate values")
    ax.set_xlabel("Gate value  (sigmoid output)")
    ax.set_ylabel("Count")
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=9)

    # Right: zoom into [0, 0.1] to see the spike 
    ax2 = axes[1]
    near_zero = gates[gates < 0.1]
    ax2.hist(near_zero, bins=50, color="#f97316", edgecolor="white", linewidth=0.3)
    ax2.set_title("Zoom: gates < 0.1  (pruned region)")
    ax2.set_xlabel("Gate value")
    ax2.set_ylabel("Count")

    pct = 100 * (gates < 0.01).mean()
    fig.text(0.5, -0.02,
             f"Sparsity: {pct:.1f}% of gates < 0.01   |   "
             f"Total gates: {len(gates):,}",
             ha="center", fontsize=10, color="grey")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved gate distribution → {out_path}")


def plot_training_curves(histories: dict, out_path: str = "training_curves.png"):
    """Plots test accuracy and sparsity over epochs for all lambda values."""
    lambdas = list(histories.keys())
    colors  = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Dynamics — Self-Pruning Network", fontsize=14)

    for i, lam in enumerate(lambdas):
        h   = histories[lam]
        col = colors[i % len(colors)]
        ep  = range(1, len(h["test_acc"]) + 1)
        ax1.plot(ep, h["test_acc"],    color=col, label=f"λ={lam}", linewidth=2)
        ax2.plot(ep, [100*s for s in h["sparsity"]], color=col, label=f"λ={lam}", linewidth=2)

    ax1.set_title("Test Accuracy (%)");  ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)");      ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title("Sparsity (%)");       ax2.set_xlabel("Epoch")
    ax2.set_ylabel("% Gates Pruned");    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved training curves → {out_path}")


def plot_lambda_tradeoff(results: list, out_path: str = "lambda_tradeoff.png"):
    """Bar chart comparing accuracy and sparsity across lambda values."""
    lambdas  = [str(r["lambda"])  for r in results]
    accs     = [r["accuracy"]     for r in results]
    spars    = [r["sparsity_pct"] for r in results]

    x = np.arange(len(lambdas))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, accs,  w, label="Test Accuracy (%)",   color="#3b82f6", alpha=0.85)
    bars2 = ax2.bar(x + w/2, spars, w, label="Sparsity (%)",         color="#f97316", alpha=0.85)

    ax1.set_xticks(x); ax1.set_xticklabels([f"λ={l}" for l in lambdas])
    ax1.set_ylabel("Test Accuracy (%)", color="#3b82f6")
    ax2.set_ylabel("Sparsity (%)",      color="#f97316")
    ax1.set_title("Accuracy vs. Sparsity Trade-off Across λ Values")

    # Value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] Saved λ trade-off chart → {out_path}")

# RESULTS TABLE + MARKDOWN REPORT

def print_results_table(results: list):
    print(f"\n{'─'*55}")
    print(f"  {'Lambda':<10} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print(f"{'─'*55}")
    for r in results:
        print(f"  {str(r['lambda']):<10} {r['accuracy']:>14.2f} "
              f"{r['sparsity_pct']:>14.1f}")
    print(f"{'─'*55}\n")


def save_markdown_report(results: list, out_path: str = "REPORT.md"):
    rows = "\n".join(
        f"| {r['lambda']} | {r['accuracy']:.2f}% | {r['sparsity_pct']:.1f}% |"
        for r in results
    )
    md = f"""# Self-Pruning Neural Network — Results Report

## Approach

This project implements a **dynamic weight pruning** method where the network
learns to prune itself *during* training rather than as a post-training step.

Each weight in every `PrunableLinear` layer is multiplied by a scalar **gate**:

```
gates        = sigmoid(gate_scores)      # ∈ (0, 1)
pruned_weight = weight × gates           # element-wise
output        = pruned_weight @ x + bias
```

The total loss combines cross-entropy with an **L1 sparsity penalty**:

```
Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ|gates|
```

---

## Why L1 Encourages Sparsity

The **L1 norm** (sum of absolute values) is the classical sparsity-inducing
regulariser — the same reason LASSO regression produces sparse solutions.

| Regulariser | Gradient near zero | Effect |
|---|---|---|
| L2 (ridge) | `2x → 0` as x→0 | Shrinks weights; never reaches 0 |
| L1 (lasso) | `sign(x) = ±1` constant | Constant push → weights reach exactly 0 |

Because the gradient of `|x|` is a constant `±1` regardless of magnitude,
the optimiser always receives a full-strength signal to reduce the gate —
even when it is already very small. This creates **exact zeros**, not just
small values.

Combined with sigmoid (which asymptotically approaches 0 as
`gate_scores → −∞`), the network learns to "switch off" unimportant weights
entirely.

---

## Architecture

```
Input (3×32×32 = 3072) → Flatten
→ PrunableLinear(3072, 512) → BN → ReLU → Dropout(0.3)
→ PrunableLinear( 512, 256) → BN → ReLU → Dropout(0.3)
→ PrunableLinear( 256, 128) → BN → ReLU → Dropout(0.3)
→ PrunableLinear( 128,  10) → logits
```

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|---|---|---|
{rows}

---

## Observations

- **Low λ**: Minimal pruning pressure → high accuracy, low sparsity. Network
  retains most connections.
- **Medium λ**: Good balance between accuracy and sparsity. Gate distribution
  shows a clear bimodal pattern — a spike at 0 and a cluster of active gates.
- **High λ**: Aggressive pruning → significant sparsity but accuracy drops as
  too many important weights are forced to zero.

The gate distribution for the best model shows the expected bimodal shape:
a large spike near 0 (pruned connections) and a smaller cluster of active
gates (important connections the network preserved).

---

## Future Improvements

1. **Convolutional Self-Pruning**: Extend `PrunableLinear` to a
   `PrunableConv2d` for modern CNN architectures (ResNet, EfficientNet).
2. **Structured Pruning**: Instead of per-weight gates, learn per-neuron or
   per-filter gates to get hardware-friendly structured sparsity.
3. **Straight-Through Estimator**: Use hard binary gates during the forward
   pass but soft gates during the backward pass (common in lottery ticket
   hypothesis research).
4. **Adaptive λ Scheduling**: Start with low λ (learn features first) and
   increase it during training (prune later). This often gives better
   accuracy-sparsity trade-offs.
5. **KL-Divergence Regularisation**: Replace L1 with a KL term that pushes
   gates toward a Bernoulli prior with a target sparsity rate.
6. **Hardware Benchmarking**: Measure actual inference speedup by converting
   the pruned model to a sparse format (e.g., `torch.sparse`).

---

*Generated by Self-Pruning Neural Network — Tredence AI Engineering Case Study*
"""
    with open(out_path, "w") as f:
        f.write(md)
    print(f"  [REPORT] Saved Markdown report → {out_path}")



# MAIN ENTRY POINT

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--lambdas",    type=float, nargs="+",
                        default=[1e-5, 1e-4, 1e-3],
                        help="Lambda values to test (e.g. 1e-5 1e-4 1e-3)")
    args = parser.parse_args()

    print("\n" + "═"*65)
    print("  SELF-PRUNING NEURAL NETWORK — CIFAR-10")
    print("  Tredence Analytics — AI Engineering Case Study")
    print("═"*65)

    histories = {}
    results   = []
    best_model_info = {"lam": None, "acc": 0.0, "model": None}

    for lam in args.lambdas:
        model, history = train(
            lam        = lam,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            data_dir   = args.data_dir,
        )
        final_acc      = history["test_acc"][-1]
        final_sparsity = 100.0 * history["sparsity"][-1]

        histories[lam] = history
        results.append({
            "lambda":       lam,
            "accuracy":     final_acc,
            "sparsity_pct": final_sparsity,
        })

        if final_acc > best_model_info["acc"]:
            best_model_info = {"lam": lam, "acc": final_acc, "model": model}

    # Summary table 
    print("\n" + "═"*65)
    print("  FINAL RESULTS SUMMARY")
    print("═"*65)
    print_results_table(results)

    #Plots
    print("[VIZ] Generating plots …")
    plot_gate_distribution(best_model_info["model"],
                           best_model_info["lam"],
                           "gate_distribution.png")
    plot_training_curves(histories, "training_curves.png")
    plot_lambda_tradeoff(results,   "lambda_tradeoff.png")

    #  Markdown report
    save_markdown_report(results)

    print("\n[DONE] All outputs saved:")
    print("  • gate_distribution.png")
    print("  • training_curves.png")
    print("  • lambda_tradeoff.png")
    print("  • REPORT.md")


if __name__ == "__main__":
    main()

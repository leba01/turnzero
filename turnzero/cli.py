"""TurnZero CLI entry point.

Commands are registered as they are implemented across pipeline tasks.
"""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """TurnZero -- turn-zero OTS coach for Pokemon VGC Gen 9."""


@cli.command()
@click.option("--raw_path", required=True, type=click.Path(exists=True),
              help="Path to raw JSON battle log file.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for parsed JSONL + manifest.")
@click.option("--limit", default=None, type=int,
              help="Only parse the first N battles (for testing).")
def parse(raw_path: str, out_dir: str, limit: int | None) -> None:
    """Parse raw Showdown logs into directed MatchExample JSONL."""
    from turnzero.data.parser import run_parse
    run_parse(raw_path, out_dir, limit=limit)


@cli.command()
@click.option("--in_path", required=True, type=click.Path(exists=True),
              help="Path to parsed match_examples.jsonl.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for canonicalized JSONL + manifest.")
def canonicalize(in_path: str, out_dir: str) -> None:
    """Canonicalize parsed examples: normalize names, sort, dedup."""
    from turnzero.data.canonicalize import run_canonicalize
    run_canonicalize(in_path, out_dir)


@cli.command()
@click.option("--in_path", required=True, type=click.Path(exists=True),
              help="Path to canonical match_examples.jsonl.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for cluster assignments + manifest.")
def cluster(in_path: str, out_dir: str) -> None:
    """Core-cluster teams by species overlap (>=4/6 species)."""
    from turnzero.splits.cluster import run_cluster
    run_cluster(in_path, out_dir)


@cli.command()
@click.option("--in_path", required=True, type=click.Path(exists=True),
              help="Path to canonical match_examples.jsonl.")
@click.option("--clusters", required=True, type=click.Path(exists=True),
              help="Path to cluster_assignments.json.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for splits.json + manifest.")
@click.option("--seed", default=42, type=int,
              help="Random seed for reproducibility.")
def split(in_path: str, clusters: str, out_dir: str, seed: int) -> None:
    """Generate train/val/test splits (Regime A + Regime B)."""
    from turnzero.splits.split import run_split
    run_split(in_path, clusters, out_dir, seed=seed)


@cli.command()
@click.option("--canonical_path", required=True, type=click.Path(exists=True),
              help="Path to canonical match_examples.jsonl.")
@click.option("--clusters", required=True, type=click.Path(exists=True),
              help="Path to cluster_assignments.json.")
@click.option("--splits", required=True, type=click.Path(exists=True),
              help="Path to splits.json.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for per-split JSONL files + manifest.")
def assemble(canonical_path: str, clusters: str, splits: str, out_dir: str) -> None:
    """Assemble per-split JSONL files with cluster IDs attached."""
    from turnzero.data.assemble import run_assemble
    run_assemble(canonical_path, clusters, splits, out_dir)


@cli.command()
@click.option("--data_dir", required=True, type=click.Path(exists=True),
              help="Path to assembled data directory.")
@click.option("--validate/--no-validate", default=True,
              help="Run integrity validation assertions.")
def stats(data_dir: str, validate: bool) -> None:
    """Comprehensive dataset stats + integrity validation."""
    from turnzero.data.stats import run_stats
    run_stats(data_dir, validate=validate)


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True),
              help="Path to YAML config file.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for checkpoints, curves, and metadata.")
def train(config: str, out_dir: str) -> None:
    """Train OTSTransformer model."""
    from turnzero.models.train import run_train
    run_train(config, out_dir)


@cli.command("eval")
@click.option("--model_ckpt", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint (best.pt).")
@click.option("--test_split", required=True, type=click.Path(exists=True),
              help="Path to test.jsonl.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for metrics and plots.")
def eval_cmd(model_ckpt: str, test_split: str, out_dir: str) -> None:
    """Evaluate a trained model on a test split."""
    from turnzero.models.train import run_eval
    run_eval(model_ckpt, test_split, out_dir)


@cli.command("calibrate")
@click.option("--model_ckpt", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint (best.pt).")
@click.option("--val_split", required=True, type=click.Path(exists=True),
              help="Path to val.jsonl for fitting temperature.")
@click.option("--out_dir", required=True, type=click.Path(),
              help="Output directory for temperature.json and calibration report.")
def calibrate_cmd(model_ckpt: str, val_split: str, out_dir: str) -> None:
    """Fit temperature scaling on validation set."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import torch

    from turnzero.data.dataset import VGCDataset, Vocab
    from turnzero.models.transformer import ModelConfig, OTSTransformer
    from turnzero.uq.temperature import TemperatureScaler, collect_logits

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Device: {device}")

    # --- Load checkpoint ---
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = OTSTransformer(ckpt["vocab_sizes"], model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Load vocab ---
    ckpt_dir = Path(model_ckpt).parent
    vocab_path = ckpt_dir / "vocab.json"
    if not vocab_path.exists():
        vocab_path = Path(ckpt["config"]["data"]["split_dir"]) / "vocab.json"
    vocab = Vocab.load(vocab_path)

    # --- Build val loader ---
    from torch.utils.data import DataLoader

    val_ds = VGCDataset(val_split, vocab)
    batch_size = ckpt["config"]["training"]["batch_size"]
    num_workers = ckpt["config"]["training"]["num_workers"]
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    click.echo(f"Val: {len(val_ds):,} examples, {len(val_loader)} batches")

    # --- Collect logits ---
    click.echo("Collecting logits ...")
    logits, labels_dict = collect_logits(model, val_loader, device)
    click.echo(f"Logits shape: {logits.shape}")

    # --- Fit temperature on Tier 1 subset (bring4_observed) ---
    tier1 = labels_dict["bring4_observed"].astype(bool)
    logits_t1 = logits[tier1]
    labels_t1 = labels_dict["action90_true"][tier1]
    click.echo(f"Tier 1 examples for fitting: {tier1.sum():,}")

    scaler = TemperatureScaler()
    report = scaler.fit(logits_t1, labels_t1)

    # --- Save artifacts ---
    scaler.save(out_path / "temperature.json")

    report_path = out_path / "calibration_report.json"
    with open(report_path, "w") as f:
        _json.dump(report, f, indent=2)

    # --- Print summary ---
    click.echo(f"\n{'─' * 50}")
    click.echo(f"Fitted temperature:  T = {report['T']:.4f}")
    click.echo(f"Val NLL  before/after: {report['val_nll_before']:.4f} → {report['val_nll_after']:.4f}")
    click.echo(f"Val ECE  before/after: {report['val_ece_before']:.4f} → {report['val_ece_after']:.4f}")
    click.echo(f"{'─' * 50}")
    click.echo(f"Saved temperature.json to {out_path / 'temperature.json'}")
    click.echo(f"Saved calibration_report.json to {report_path}")


@cli.command("demo")
@click.option("--ensemble_dir", required=True, type=click.Path(exists=True),
              help="Path to directory containing ensemble_001..005/ subdirs.")
@click.option("--calib", default=None, type=click.Path(exists=True),
              help="Path to temperature.json (optional; T=1.0 if omitted).")
@click.option("--team_a", default=None, type=str,
              help="Comma-separated species list for Team A (your team).")
@click.option("--team_b", default=None, type=str,
              help="Comma-separated species list for Team B (opponent).")
@click.option("--team_a_ots", default=None, type=click.Path(exists=True),
              help="Path to Team A OTS JSON file (alternative to --team_a).")
@click.option("--team_b_ots", default=None, type=click.Path(exists=True),
              help="Path to Team B OTS JSON file (alternative to --team_b).")
@click.option("--vocab", default=None, type=click.Path(exists=True),
              help="Path to vocab.json (defaults to ensemble_001/vocab.json).")
@click.option("--tau", default=0.04, type=float,
              help="Abstention threshold for confidence (default 0.04).")
@click.option("--top_k", default=3, type=int,
              help="Number of top plans to display (default 3).")
@click.option("--index_path", default=None, type=click.Path(),
              help="Path to pre-built RetrievalIndex (base path, no extension).")
@click.option("--retrieval_k", default=10, type=int,
              help="Number of retrieval neighbors to query (default 10).")
def demo_cmd(ensemble_dir, calib, team_a, team_b, team_a_ots, team_b_ots,
             vocab, tau, top_k, index_path, retrieval_k):
    """Run the turn-zero coach demo: predict top plans for a matchup."""
    from turnzero.tool.coach import run_demo

    if not team_a and not team_a_ots:
        raise click.UsageError("Must provide either --team_a or --team_a_ots")
    if not team_b and not team_b_ots:
        raise click.UsageError("Must provide either --team_b or --team_b_ots")

    run_demo(
        ensemble_dir=ensemble_dir,
        calib_path=calib,
        team_a_str=team_a,
        team_b_str=team_b,
        team_a_ots=team_a_ots,
        team_b_ots=team_b_ots,
        vocab_path=vocab,
        tau=tau,
        top_k=top_k,
        index_path=index_path,
        retrieval_k=retrieval_k,
    )


if __name__ == "__main__":
    cli()

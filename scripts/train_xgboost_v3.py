"""
"""
Train XGBoost v3 with temporal split and fee-adjusted targets.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from analysis.simple_threshold_backtest import evaluate_threshold
from backtesting.honest_backtest_v2 import HonestBacktestV2
from core.feature_engine_v2 import FeatureSpecV3
from core.feature_loader import FeatureLoader

logger = logging.getLogger(__name__)

MIN_TRAIN_SAMPLES = 1000
MIN_VAL_SAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost v3 model")
    parser.add_argument("--config-path", type=str, default="configs/model2_training.yaml")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--no-backtest", action="store_true", help="Skip validation backtest and threshold search")
    parser.add_argument("--verbose", action="store_true", help="Verbose diagnostics (probability stats, etc.)")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def sanitize_features_and_target(
    X: pd.DataFrame,
    y: pd.Series,
    context: str = "train",
    max_bad_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove rows with NaN/Inf and enforce a bad-row budget."""

    X_checked = X.replace([np.inf, -np.inf], np.nan)
    y_checked = y.replace([np.inf, -np.inf], np.nan)

    good_mask = ~(X_checked.isna().any(axis=1) | y_checked.isna())
    bad_ratio = 1.0 - float(good_mask.mean())

    if bad_ratio > max_bad_ratio:
        raise ValueError(
            f"[sanitize_features] {context}: bad ratio {bad_ratio:.2%} exceeds the threshold {max_bad_ratio:.2%}"
        )

    return X.loc[good_mask], y.loc[good_mask]


def compute_targets(df: pd.DataFrame, horizon: int, commission: float, slippage: float, target_type: str) -> pd.DataFrame:
    df = df.copy()
    entry_price = df["open"].shift(-1)
    exit_price = df["close"].shift(-horizon)

    raw_ret = (exit_price - entry_price) / entry_price
    round_trip_cost = (commission + slippage) * 2
    fee_ret = raw_ret - round_trip_cost

    df["raw_ret"] = raw_ret
    df["fee_ret"] = fee_ret
    df["rapnl"] = fee_ret  # Placeholder until RAPnL is formalised

    if target_type == "fee_ret":
        target_series = (df["fee_ret"] > 0).astype(int)
    elif target_type == "raw_ret":
        target_series = (df["raw_ret"] > 0).astype(int)
    else:
        target_series = (df["rapnl"] > 0).astype(int)

    df["target"] = target_series
    df = df.iloc[:-horizon]  # drop tail with NaNs/unknown targets

    return df.dropna(subset=["target"])


def temporal_split(df: pd.DataFrame, train_start: str, train_end: str, val_end: str, test_end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_index()

    train_mask = (df_sorted.index >= pd.Timestamp(train_start)) & (df_sorted.index < pd.Timestamp(train_end))
    val_mask = (df_sorted.index >= pd.Timestamp(train_end)) & (df_sorted.index < pd.Timestamp(val_end))
    if test_end:
        test_mask = (df_sorted.index >= pd.Timestamp(val_end)) & (df_sorted.index < pd.Timestamp(test_end))
    else:
        test_mask = df_sorted.index >= pd.Timestamp(val_end)

    return df_sorted[train_mask], df_sorted[val_mask], df_sorted[test_mask]


def build_dmatrices(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, spec: FeatureSpecV3):
    X_train = spec.enforce(train_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_val = spec.enforce(val_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_test = spec.enforce(test_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> xgb.XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    params = params.copy()
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("scale_pos_weight", scale_pos_weight)
    params.setdefault("random_state", 42)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "threshold": threshold,
        "samples": int(len(y)),
        "class_balance": float(y.mean()),
    }


def backtest_thresholds(probabilities: np.ndarray, val_df: pd.DataFrame, thresholds: List[float], backtest_cfg: Dict) -> Dict:
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv = val_df[ohlcv_cols]
    results = {}

    for th in thresholds:
        signals = (probabilities >= th).astype(int)
        engine = HonestBacktestV2(
            commission=backtest_cfg.get("commission", 0.001),
            slippage=backtest_cfg.get("slippage", 0.0005),
            max_hold_bars=backtest_cfg.get("max_hold_bars", 100),
            take_profit=backtest_cfg.get("take_profit", 0.02),
            stop_loss=backtest_cfg.get("stop_loss", 0.01),
            cooldown_bars=backtest_cfg.get("cooldown_bars", 10),
        )
        metrics = engine.run(ohlcv, signals)
        results[th] = metrics

    return results


def select_optimal_threshold(backtest_results: Dict[float, Dict], max_dd_limit: float) -> Dict:
    best_th = None
    best_sharpe = -np.inf
    for th, metrics in backtest_results.items():
        if metrics.get("max_drawdown_pct", 0) <= max_dd_limit and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_th = th
    all_sharpes = [m.get("sharpe_ratio", -np.inf) for m in backtest_results.values()]
    no_profitable = all(s <= 0 for s in all_sharpes)
    return {
        "threshold": best_th,
        "sharpe": best_sharpe,
        "backtest_metrics": backtest_results.get(best_th, {}),
        "no_profitable_threshold": no_profitable,
    }


def run_threshold_grid_search(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Run grid search over thresholds and return full metrics table.
    """

    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    rows = []
    for thr in thresholds:
        metrics = evaluate_threshold(df, proba_col, fee_ret_col, thr)
        rows.append(metrics)

    return pd.DataFrame(rows)


def export_val_trades(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    coin: str,
    tf: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Export validation trades (where signal=1) to CSV for manual analysis.
    """

    if output_dir is None:
        output_dir = Path("reports")

    output_dir.mkdir(exist_ok=True)

    df = val_df.copy()
    df["P_ml"] = proba
    df["signal"] = (df["P_ml"] >= threshold).astype(int)

    trades_df = df[df["signal"] == 1].copy()

    export_cols: List[str] = []
    for col in ["open", "high", "low", "close", "volume"]:
        if col in trades_df.columns:
            export_cols.append(col)

    for col in ["P_ml", "fee_ret", "raw_ret", "signal"]:
        if col in trades_df.columns and col not in export_cols:
            export_cols.append(col)

    if not export_cols:
        export_cols = list(trades_df.columns)

    trades_df = trades_df[export_cols]

    out_path = output_dir / f"model2_val_trades_{coin.lower()}_{tf}.csv"
    trades_df.to_csv(out_path)

    return out_path


def save_model(model: xgb.XGBClassifier, output_path: Path, feature_spec: FeatureSpecV3, metadata: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = metadata.copy()
    metadata.update({
        "feature_names": feature_spec.feature_names,
        "n_features": feature_spec.n_features,
    })
    meta_path.write_text(json.dumps(metadata, indent=2))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config_path)

    model_cfg = cfg["model2"]
    costs_cfg = cfg.get("costs", {})
    backtest_cfg = cfg.get("backtest", {})

    loader = FeatureLoader(data_dir=model_cfg.get("data_dir", "data/features"))
    coin = model_cfg.get("coin", "BTC")
    timeframe = model_cfg.get("timeframe", "15m")
    file_path = loader.get_file_path(coin, timeframe)
    df = loader.load_features(coin=coin, timeframe=timeframe, start_date=model_cfg["train_start"], end_date=model_cfg.get("test_end"), validate=True)

    df_with_targets = compute_targets(
        df,
        horizon=model_cfg["horizon_bars"],
        commission=costs_cfg.get("commission", 0.001),
        slippage=costs_cfg.get("slippage", 0.0005),
        target_type=model_cfg.get("target_type", "fee_ret"),
    )

    train_df, val_df, test_df = temporal_split(
        df_with_targets,
        train_start=model_cfg["train_start"],
        train_end=model_cfg["train_end"],
        val_end=model_cfg["val_end"],
        test_end=model_cfg.get("test_end"),
    )

    if len(train_df) < MIN_TRAIN_SAMPLES or len(val_df) < MIN_VAL_SAMPLES:
        logger.warning(f"SKIPPED: insufficient data for {coin}/{timeframe}")
        logger.warning(f"  Train: {len(train_df)} (need >= {MIN_TRAIN_SAMPLES})")
        logger.warning(f"  Val:   {len(val_df)} (need >= {MIN_VAL_SAMPLES})")
        logger.warning(f"  Data range: {df.index.min()} to {df.index.max()}")

        metadata = {
            "status": "SKIPPED",
            "reason": "insufficient_history",
            "coin": coin,
            "timeframe": timeframe,
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "data_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        metadata_path = output_dir / f"xgboost_v3_{coin.lower()}_{timeframe}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel SKIPPED: {coin}/{timeframe} (insufficient history)")
        print(f"Metadata saved to: {metadata_path}")
        sys.exit(0)

    feature_spec = FeatureSpecV3.from_dataframe(train_df)
    if feature_spec.n_features != 74:
        raise ValueError(f"Expected 74 features, got {feature_spec.n_features}")

    X_train, X_val, X_test, y_train, y_val, y_test = build_dmatrices(train_df, val_df, test_df, feature_spec)

    X_train, y_train = sanitize_features_and_target(X_train, y_train, context="train")
    train_df = train_df.loc[X_train.index]

    X_val, y_val = sanitize_features_and_target(X_val, y_val, context="val")
    val_df = val_df.loc[X_val.index]

    X_test, y_test = sanitize_features_and_target(X_test, y_test, context="test")
    test_df = test_df.loc[X_test.index]

    unique_train = pd.unique(y_train)
    unique_val = pd.unique(y_val)

    degenerate_reasons = []
    if len(unique_train) < 2:
        degenerate_reasons.append("degenerate_labels_train")
    if len(unique_val) < 2:
        degenerate_reasons.append("degenerate_labels_val")

    if degenerate_reasons:
        reason = ",".join(degenerate_reasons)
        logger.warning(f"SKIPPED: degenerate labels for {coin}/{timeframe} ({reason})")
        logger.warning(f"  y_train unique: {unique_train}")
        logger.warning(f"  y_val   unique: {unique_val}")

        metadata = {
            "status": "SKIPPED",
            "reason": reason,
            "coin": coin,
            "timeframe": timeframe,
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "label_stats": {
                "train_unique": [int(v) for v in unique_train],
                "val_unique": [int(v) for v in unique_val],
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        metadata_path = output_dir / f"xgboost_v3_{coin.lower()}_{timeframe}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nModel SKIPPED: {coin}/{timeframe} (degenerate labels)")
        print(f"Metadata saved to: {metadata_path}")
        sys.exit(0)

    model_params = model_cfg.get("xgboost_params", {})
    model = train_model(X_train, y_train, X_val, y_val, model_params)

    metrics_train = evaluate_model(model, X_train, y_train, threshold=0.5)
    metrics_val = evaluate_model(model, X_val, y_val, threshold=0.5)
    metrics_test = evaluate_model(model, X_test, y_test, threshold=0.5)

    optimal = {"threshold": None, "sharpe": None, "backtest_metrics": {}}
    val_trades_path: Optional[Path] = None
    if not args.no_backtest:
        print("Optimizing threshold on validation set...")
        val_prob = model.predict_proba(X_val)[:, 1]

        val_df_clean = val_df.loc[X_val.index].copy()
        val_df_clean["P_ml"] = val_prob

        if args.verbose:
            print("\nProbability distribution on Val:")
            print(f"  Min:    {val_prob.min():.4f}")
            print(f"  Max:    {val_prob.max():.4f}")
            print(f"  Mean:   {val_prob.mean():.4f}")
            print(f"  Median: {np.median(val_prob):.4f}")
            for thr in [0.3, 0.5, 0.7]:
                count = (val_prob > thr).sum()
                frac = count / len(val_prob) * 100.0
                print(f"  > {thr:.1f}: {count} samples ({frac:.2f}%)")
            print()

        if args.verbose:
            print("\n" + "=" * 60)
            print("THRESHOLD GRID SEARCH (validation)")
            print("=" * 60)

        grid_df = run_threshold_grid_search(
            val_df_clean,
            proba_col="P_ml",
            fee_ret_col="fee_ret",
            thresholds=backtest_cfg.get("threshold_grid"),
        )

        if args.verbose:
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(grid_df.to_string(index=False))

            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            grid_path = reports_dir / f"threshold_grid_{coin.lower()}_{timeframe}.csv"
            grid_df.to_csv(grid_path, index=False)
            print(f"\nThreshold grid saved to: {grid_path}")

        thresholds = backtest_cfg.get("threshold_grid", [round(x, 2) for x in np.linspace(0.5, 0.9, 5)])
        bt_results = backtest_thresholds(val_prob, val_df, thresholds, backtest_cfg)
        optimal = select_optimal_threshold(bt_results, max_dd_limit=backtest_cfg.get("max_dd_pct", 20))

        best_thr = optimal["threshold"]
        bt_metrics = optimal.get("backtest_metrics", {})
        n_trades = bt_metrics.get("n_trades", 0)

        val_trades_path = None
        if best_thr is not None and n_trades > 0:
            val_trades_path = export_val_trades(
                val_df=val_df_clean,
                proba=val_prob,
                threshold=best_thr,
                coin=coin,
                tf=timeframe,
                output_dir=Path(backtest_cfg.get("diagnostics_dir", "diagnostics")),
            )
            print(f"Exported {n_trades} validation trades to: {val_trades_path}")

    output_path = Path(model_cfg.get("output_path", f"models/xgboost_v3_{coin.lower()}_{timeframe}.json"))
    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "coin": coin,
        "timeframe": timeframe,
        "experiment": args.experiment_name,
        "source_data": str(file_path),
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "optimal_threshold_trading": optimal,
        "target_type": model_cfg.get("target_type", "fee_ret"),
        "horizon_bars": model_cfg.get("horizon_bars"),
        "val_trades_path": str(val_trades_path) if val_trades_path else None,
    }

    save_model(model, output_path, feature_spec, metadata)

    summary = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "best_threshold": optimal,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

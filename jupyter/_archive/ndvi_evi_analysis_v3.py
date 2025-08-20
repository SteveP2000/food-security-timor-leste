# ===============================
# Analysis notebook (cleaned)
# ===============================
from pathlib import Path
import glob
import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------- 1) Paths & config ----------
INPUT_DIR  = Path(r"C:\temp\timor_leste\ndvi_evi")          # monthly ADM2 CSVs
OUTPUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")  # all outputs
PLOTS_DIR  = OUTPUT_DIR / "plots"

# Plot subfolders
PLOT_DIR_AUC_NDVI        = PLOTS_DIR / "auc_ndvi"
PLOT_DIR_AUC_EVI         = PLOTS_DIR / "auc_evi"
PLOT_DIR_HARV_MEAN_NDVI  = PLOTS_DIR / "harvest_mean_ndvi"
PLOT_DIR_HARV_MEAN_EVI   = PLOTS_DIR / "harvest_mean_evi"

# Seasonal overlay / time-series dirs (estimated harvest windows)
OVERLAY_NDVI_DIR   = PLOTS_DIR / "seasonal_overlay_ndvi_estharvest"
OVERLAY_EVI_DIR    = PLOTS_DIR / "seasonal_overlay_evi_estharvest"
TS_NDVI_DIR        = PLOTS_DIR / "timeseries_ndvi_estharvest"
TS_EVI_DIR         = PLOTS_DIR / "timeseries_evi_estharvest"

# Make dirs once
for d in [OUTPUT_DIR, PLOTS_DIR, PLOT_DIR_AUC_NDVI, PLOT_DIR_AUC_EVI,
          PLOT_DIR_HARV_MEAN_NDVI, PLOT_DIR_HARV_MEAN_EVI,
          OVERLAY_NDVI_DIR, OVERLAY_EVI_DIR, TS_NDVI_DIR, TS_EVI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global analysis config
BASELINE_YEARS  = [2019, 2020, 2021]
ANALYSIS_YEARS  = [2022, 2023, 2024, 2025]
HARVEST_WIN_CSV = OUTPUT_DIR / "estimated_harvest_windows.csv"
USE_CONSENSUS_FOR_BOTH = False  # True → use consensus window for NDVI & EVI alike

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 80)

# ---------- 2) I/O helpers ----------
NUMERIC_COLS_DEFAULT = [
    "mean_NDVI","max_NDVI","mean_EVI","max_EVI",
    "clear_frac_mean","clear_frac_max","count_images"
]

def _read_one_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {fp}")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["days_in_month"] = df["date"].dt.days_in_month

    # Infer ADM2 from filename if absent
    adm2_guess = Path(fp).name.split("_")[0]
    if "ADM2_PCODE" not in df.columns:
        df["ADM2_PCODE"] = adm2_guess
    else:
        df["ADM2_PCODE"] = df["ADM2_PCODE"].fillna(adm2_guess)

    # Cast numerics
    for col in NUMERIC_COLS_DEFAULT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_monthly_files(input_dir: Path, max_workers: int = 12) -> pd.DataFrame:
    files = glob.glob(str(input_dir / "*.csv"))
    if not files:
        raise RuntimeError(f"No CSVs found in {input_dir}")
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_one_csv, fp): fp for fp in files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading CSVs"):
            try:
                frames.append(f.result())
            except Exception as e:
                print(f"Skipping {futures[f]}: {e}")
    df = pd.concat(frames, ignore_index=True)

    # ---- De-duplicate per ADM2×date (averages numerics) to prevent double counting
    keys = ["ADM2_PCODE","date"]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "mean" for c in num_cols}
    for k in keys:
        agg[k] = "first"
    df = (df.groupby(keys, as_index=False).agg(agg)
            .sort_values(keys).reset_index(drop=True))
    # Recompute calendar fields (robust)
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["days_in_month"] = df["date"].dt.days_in_month
    return df

# ---------- 3) Calendar & general helpers ----------
def padded_limits(series: pd.Series, pad: float = 0.05):
    s = series.dropna()
    if s.empty: return (0, 1)
    lo, hi = s.min(), s.max()
    if lo == hi:
        span = abs(hi) if hi != 0 else 1.0
        lo, hi = hi - 0.1*span, hi + 0.1*span
    pad_span = (hi - lo) * pad
    return (lo - pad_span, hi + pad_span)

def slope_per_adm2(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for adm2, sub in df[["ADM2_PCODE","year",value_col]].dropna().groupby("ADM2_PCODE"):
        x = sub["year"].values.astype(float)
        y = sub[value_col].values.astype(float)
        slope = np.polyfit(x, y, 1)[0] if len(np.unique(x)) >= 2 else np.nan
        rows.append({"ADM2_PCODE": adm2, f"slope_{value_col}": slope})
    return pd.DataFrame(rows)

# ---------- 4) AUC helpers (no groupby.apply warnings) ----------
def annual_auc(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    cols = ["ADM2_PCODE", "year", "days_in_month", index_col]
    tmp = df.loc[:, cols].copy()
    tmp[index_col] = pd.to_numeric(tmp[index_col], errors="coerce")
    tmp["days_in_month"] = pd.to_numeric(tmp["days_in_month"], errors="coerce")
    tmp["w"] = tmp[index_col] * tmp["days_in_month"]
    out = (tmp.groupby(["ADM2_PCODE","year"], as_index=False)["w"].sum()
              .rename(columns={"w": f"AUC_{index_col}"}))
    return out

def annual_auc_clearweighted(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    if "clear_frac_mean" not in df.columns:
        return pd.DataFrame(columns=["ADM2_PCODE","year",f"AUCcw_{index_col}"])
    cols = ["ADM2_PCODE","year","days_in_month",index_col,"clear_frac_mean"]
    tmp = df.loc[:, cols].copy()
    tmp[index_col] = pd.to_numeric(tmp[index_col], errors="coerce")
    tmp["days_in_month"] = pd.to_numeric(tmp["days_in_month"], errors="coerce")
    tmp["clear_frac_mean"] = pd.to_numeric(tmp["clear_frac_mean"], errors="coerce").fillna(0).clip(0,1)
    tmp["w"] = tmp[index_col] * tmp["days_in_month"] * tmp["clear_frac_mean"]
    out = (tmp.groupby(["ADM2_PCODE","year"], as_index=False)["w"].sum()
              .rename(columns={"w": f"AUCcw_{index_col}"}))
    return out

# ---------- 5) Estimated harvest windows ----------
def _months_in_span(start_m: int, end_m: int) -> set:
    if pd.isna(start_m) or pd.isna(end_m): return set()
    s, e = int(start_m), int(end_m)
    if s <= e: return set(range(s, e+1))
    return set(list(range(s, 13)) + list(range(1, e+1)))

def _cyclic_roll(arr, shift):
    shift = shift % len(arr)
    return np.concatenate([arr[shift:], arr[:shift]]) if shift else arr.copy()

def _smooth_cyclic(vals, window=3):
    assert len(vals) == 12
    padded = np.r_[vals[-(window//2):], vals, vals[:(window//2)]]
    return np.convolve(padded, np.ones(window)/window, mode='valid')

def _interp_cyclic(months, vals):
    m = np.asarray(months, dtype=float)
    v = np.asarray(vals, dtype=float)
    isnan = np.isnan(v)
    if isnan.all(): return v
    m_ext = np.r_[0, m, 13]
    v_ext = np.r_[v[11], v, v[0]]
    good = ~np.isnan(v_ext)
    v_interp = np.interp(m_ext, m_ext[good], v_ext[good])
    return v_interp[1:-1]

def _build_recent_climatology(df_adm2, value_col):
    months = np.arange(1, 13, dtype=int)
    yrs = sorted(df_adm2["year"].unique().tolist())
    yrs_recent = [y for y in ANALYSIS_YEARS if y in yrs]
    use_years = yrs_recent if len(yrs_recent) >= 2 else yrs
    piv = (df_adm2[df_adm2["year"].isin(use_years)]
           .pivot_table(index="month", values=value_col, aggfunc="mean"))
    return months, piv.reindex(months).values.ravel()

def _build_recent_clear(df_adm2):
    months = np.arange(1, 13, dtype=int)
    if "clear_frac_mean" not in df_adm2.columns:
        return months, np.full(12, np.nan)
    piv = (df_adm2.pivot_table(index="month", values="clear_frac_mean", aggfunc="mean"))
    return months, piv.reindex(months).values.ravel()

def _estimate_window_from_series(months, vals, clear=None, min_clear=0.20):
    months = np.asarray(months, dtype=int)
    x = np.asarray(vals, dtype=float)
    if clear is not None:
        c = np.asarray(clear, dtype=float)
        x = np.where((~np.isnan(c)) & (c < min_clear), np.nan, x)
    x_filled = _interp_cyclic(months, x)
    x_sm = _smooth_cyclic(x_filled, 3)

    peak_idx = int(np.nanargmax(x_sm))
    peak_month = int(months[peak_idx])

    rolled = _cyclic_roll(x_sm, peak_idx)
    search_slice = rolled[1:7]
    trough_rel = int(np.nanargmin(search_slice)) + 1
    trough_idx = (peak_idx + trough_rel) % 12
    trough_month = int(months[trough_idx])

    peak = x_sm[peak_idx]; trough = x_sm[trough_idx]
    R = float(peak - trough)
    T1 = peak - 0.20 * R; T2 = peak - 0.60 * R

    start_idx = end_idx = None
    for k in range(1, 7):
        idx = (peak_idx + k) % 12
        if start_idx is None and x_sm[idx] <= T1:
            start_idx = idx
        if start_idx is not None and x_sm[idx] <= T2:
            end_idx = idx; break

    if start_idx is None or end_idx is None:
        diffs = np.r_[np.diff(x_sm), x_sm[0]-x_sm[-1]]
        idxs = [(peak_idx + k) % 12 for k in range(1, 7)]
        neg_slopes = diffs[idxs]
        center_rel = int(np.nanargmin(neg_slopes)) + 1
        center_idx = (peak_idx + center_rel) % 12
        start_idx = start_idx or center_idx
        end_idx   = end_idx   or ((center_idx + 1) % 12)
        method = "steepest_decline_fallback"
    else:
        method = "percent_of_range"

    start_month = int(months[start_idx]); end_month = int(months[end_idx])

    seq = []
    i = start_idx
    while True:
        j = (i + 1) % 12
        seq.append(x_sm[j] - x_sm[i])
        i = j
        if i == end_idx: break
    monotone = float(np.mean(np.array(seq) < 0)) if seq else 0.0
    conf = float(np.clip((R / 0.35) * (0.5 + 0.5 * monotone), 0, 1))

    return dict(
        start_month=start_month, end_month=end_month,
        peak_month=peak_month, trough_month=trough_month,
        range_R=round(R,3), confidence=round(conf,3), method=method
    )

def estimate_harvest_windows(data: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per ADM2 with NDVI/EVI+consensus windows; also writes CSV."""
    recs = []
    for adm2 in tqdm(sorted(data["ADM2_PCODE"].unique()), desc="Estimating harvest windows"):
        sub = data.loc[data["ADM2_PCODE"]==adm2,
                       ["ADM2_PCODE","year","month","mean_NDVI","mean_EVI","clear_frac_mean"]].copy()
        m_ndvi, s_ndvi = _build_recent_climatology(sub, "mean_NDVI")
        m_evi,  s_evi  = _build_recent_climatology(sub, "mean_EVI")
        m_clr,  s_clr  = _build_recent_clear(sub)

        nd = _estimate_window_from_series(m_ndvi, s_ndvi, s_clr)
        ev = _estimate_window_from_series(m_evi,  s_evi,  s_clr)

        def _span_to_set(a,b):
            return _months_in_span(a,b)

        nd_set = _span_to_set(nd["start_month"], nd["end_month"])
        ev_set = _span_to_set(ev["start_month"], ev["end_month"])
        uni    = sorted(list(nd_set | ev_set))
        if not uni:
            cons_s, cons_e, cons_m = nd["start_month"], nd["end_month"], "NDVI_only"
        else:
            cons_s, cons_e, cons_m = uni[0], uni[-1], "union(NDVI,EVI)"

        recs.append({
            "ADM2_PCODE":               adm2,
            "ndvi_start_month":         nd["start_month"],
            "ndvi_end_month":           nd["end_month"],
            "ndvi_peak_month":          nd["peak_month"],
            "ndvi_trough_month":        nd["trough_month"],
            "ndvi_range":               nd["range_R"],
            "ndvi_confidence":          nd["confidence"],
            "ndvi_method":              nd["method"],
            "evi_start_month":          ev["start_month"],
            "evi_end_month":            ev["end_month"],
            "evi_peak_month":           ev["peak_month"],
            "evi_trough_month":         ev["trough_month"],
            "evi_range":                ev["range_R"],
            "evi_confidence":           ev["confidence"],
            "evi_method":               ev["method"],
            "consensus_start_month":    cons_s,
            "consensus_end_month":      cons_e,
            "consensus_method":         cons_m
        })
    hw = pd.DataFrame.from_records(recs)
    hw.to_csv(HARVEST_WIN_CSV, index=False)
    print("Saved harvest window estimates to:", HARVEST_WIN_CSV)
    return hw

# Vectorized harvest means over estimated windows
def harvest_means_estimated(df: pd.DataFrame, index_col: str,
                            hw_df: pd.DataFrame, which: str = "ndvi") -> pd.DataFrame:
    if which not in {"ndvi","evi","consensus"}:
        raise ValueError("which must be 'ndvi','evi', or 'consensus'")
    if which == "ndvi":
        cols = ("ndvi_start_month","ndvi_end_month")
    elif which == "evi":
        cols = ("evi_start_month","evi_end_month")
    else:
        cols = ("consensus_start_month","consensus_end_month")

    # Expand windows → long table [ADM2_PCODE, month] allowed
    rows = []
    for r in hw_df.itertuples(index=False):
        mset = _months_in_span(getattr(r, cols[0]), getattr(r, cols[1]))
        rows += [(r.ADM2_PCODE, m) for m in sorted(mset)]
    allow = pd.DataFrame(rows, columns=["ADM2_PCODE","month"])
    if allow.empty:
        return pd.DataFrame(columns=["ADM2_PCODE","year",f"harv_{index_col}"])

    # Join and average
    sub = df[["ADM2_PCODE","year","month",index_col]].copy()
    sub[index_col] = pd.to_numeric(sub[index_col], errors="coerce")
    merged = sub.merge(allow, on=["ADM2_PCODE","month"], how="inner")
    out = (merged.groupby(["ADM2_PCODE","year"], as_index=False)[index_col]
                 .mean().rename(columns={index_col: f"harv_{index_col}"}))
    return out

# ---------- 6) Plots ----------
def plot_metric_by_adm2(df: pd.DataFrame, value_col: str, title: str, out_png: Path, ylim=None):
    plt.figure()
    for adm2, sub in df.sort_values(["ADM2_PCODE","year"]).groupby("ADM2_PCODE"):
        plt.plot(sub["year"], sub[value_col], label=adm2)
    plt.title(title); plt.xlabel("Year"); plt.ylabel(value_col)
    if ylim is not None: plt.ylim(ylim)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_per_adm2(df: pd.DataFrame, value_col: str, out_dir: Path, ylim=None, title_prefix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    for adm2, sub in tqdm(df.groupby("ADM2_PCODE"), total=df["ADM2_PCODE"].nunique(), desc=f"Plots: {value_col}"):
        plt.figure()
        plt.plot(sub["year"], sub[value_col])
        plt.title(f"{title_prefix}{adm2}")
        plt.xlabel("Year"); plt.ylabel(value_col)
        if ylim is not None: plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(out_dir / f"{adm2}_{value_col}.png")
        plt.close()

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _shade_window_spans(start_m, end_m):
    if pd.isna(start_m) or pd.isna(end_m): return []
    s, e = int(start_m), int(end_m)
    if 1 <= s <= 12 and 1 <= e <= 12:
        if s <= e: return [(s - 0.5, e + 0.5)]
        return [(0.5, e + 0.5), (s - 0.5, 12.5)]
    return []

def seasonal_overlay(adm2_df: pd.DataFrame, value_col: str, start_m: int, end_m: int, out_path: Path):
    months = np.arange(1, 13, dtype=int)
    piv = (adm2_df.pivot_table(index="month", columns="year", values=value_col, aggfunc="mean")
           .reindex(index=months))
    plt.figure()
    for y in sorted([c for c in piv.columns if pd.notna(c)]):
        plt.plot(months, piv[y].values, label=str(int(y)))
    base_cols = [y for y in BASELINE_YEARS if y in piv.columns]
    if base_cols:
        plt.plot(months, piv[base_cols].mean(axis=1).values, linewidth=3, label="Baseline (2019–2021)")
    for x0, x1 in _shade_window_spans(start_m, end_m):
        plt.axvspan(x0, x1, alpha=0.12)
    plt.xticks(months, MONTH_LABELS); plt.ylim(0,1)
    plt.xlabel("Month"); plt.ylabel(value_col)
    adm2 = str(adm2_df["ADM2_PCODE"].iloc[0])
    title_h = (f" (est. harvest {MONTH_LABELS[int(start_m)-1]}–{MONTH_LABELS[int(end_m)-1]})"
               if not (pd.isna(start_m) or pd.isna(end_m)) else " (no estimate)")
    metric = "NDVI" if "NDVI" in value_col else "EVI"
    plt.title(f"{adm2} — Seasonal overlay ({metric}){title_h}")
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def _eom(y, m): return datetime(y, m, calendar.monthrange(y, m)[1])

def _year_shades(y, start_m, end_m):
    if pd.isna(start_m) or pd.isna(end_m): return []
    s, e = int(start_m), int(end_m)
    spans = []
    if s <= e:
        spans.append((datetime(y, s, 1), _eom(y, e)))
    else:
        spans.append((datetime(y, 1, 1), _eom(y, e)))
        spans.append((datetime(y, s, 1), datetime(y, 12, 31)))
    return spans

def timeseries_with_shading(adm2_df: pd.DataFrame, value_col: str, start_m: int, end_m: int, out_path: Path):
    dfp = adm2_df.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(dfp["date"], dfp[value_col], marker="o", linewidth=1.5, label=value_col)
    for y in sorted(dfp["year"].unique()):
        for d0, d1 in _year_shades(y, start_m, end_m):
            left  = max(d0, dfp["date"].min().to_pydatetime())
            right = min(d1, dfp["date"].max().to_pydatetime())
            if left <= right:
                ax.axvspan(left, right, alpha=0.12)
    ax.set_ylim(0,1); ax.set_ylabel(value_col); ax.set_xlabel("Date")
    adm2 = str(dfp["ADM2_PCODE"].iloc[0])
    title_h = (f" (est. harvest {MONTH_LABELS[int(start_m)-1]}–{MONTH_LABELS[int(end_m)-1]})"
               if not (pd.isna(start_m) or pd.isna(end_m)) else " (no estimate)")
    ax.set_title(f"{adm2} — Monthly time series{title_h}")
    ax.grid(True, axis="y", alpha=0.2); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

# ---------- 7) Load monthly & estimate harvest windows ----------
data = load_monthly_files(INPUT_DIR)
print("Rows:", len(data), " | ADM2s:", data["ADM2_PCODE"].nunique())

harv_windows = estimate_harvest_windows(data)

# ---------- 8) Build annual metrics (AUCs & harvest-window means) ----------
auc_ndvi    = annual_auc(data, "mean_NDVI")
auc_evi     = annual_auc(data, "mean_EVI")
auc_ndvi_cw = annual_auc_clearweighted(data, "mean_NDVI")
auc_evi_cw  = annual_auc_clearweighted(data, "mean_EVI")

if USE_CONSENSUS_FOR_BOTH:
    harv_ndvi = harvest_means_estimated(data, "mean_NDVI", harv_windows, which="consensus")
    harv_evi  = harvest_means_estimated(data, "mean_EVI",  harv_windows, which="consensus")
else:
    harv_ndvi = harvest_means_estimated(data, "mean_NDVI", harv_windows, which="ndvi")
    harv_evi  = harvest_means_estimated(data, "mean_EVI",  harv_windows, which="evi")

metrics = (auc_ndvi.merge(auc_evi,     on=["ADM2_PCODE","year"], how="outer")
                    .merge(auc_ndvi_cw, on=["ADM2_PCODE","year"], how="left")
                    .merge(auc_evi_cw,  on=["ADM2_PCODE","year"], how="left")
                    .merge(harv_ndvi,   on=["ADM2_PCODE","year"], how="left")
                    .merge(harv_evi,    on=["ADM2_PCODE","year"], how="left"))

# Ensure uniqueness per ADM2×year (safety against accidental duplicates)
if metrics.duplicated(["ADM2_PCODE","year"]).any():
    num_cols = metrics.select_dtypes(include=[np.number]).columns.tolist()
    metrics = (metrics.groupby(["ADM2_PCODE","year"], as_index=False)[num_cols].mean())

# ---------- 9) Baseline (2019–2021) & anomalies ----------
base = (metrics[metrics["year"].isin(BASELINE_YEARS)]
        .groupby("ADM2_PCODE")
        .agg({
            "AUC_mean_NDVI":"mean","AUC_mean_EVI":"mean",
            "harv_mean_NDVI":"mean","harv_mean_EVI":"mean",
            "AUCcw_mean_NDVI":"mean","AUCcw_mean_EVI":"mean",
        }).rename(columns={
            "AUC_mean_NDVI":"base_AUC_NDVI",
            "AUC_mean_EVI":"base_AUC_EVI",
            "harv_mean_NDVI":"base_harv_NDVI",
            "harv_mean_EVI":"base_harv_EVI",
            "AUCcw_mean_NDVI":"base_AUCcw_NDVI",
            "AUCcw_mean_EVI":"base_AUCcw_EVI",
        }).reset_index())

metrics = metrics.merge(base, on="ADM2_PCODE", how="left")

for col, bcol in [
    ("AUC_mean_NDVI","base_AUC_NDVI"),
    ("AUC_mean_EVI","base_AUC_EVI"),
    ("harv_mean_NDVI","base_harv_NDVI"),
    ("harv_mean_EVI","base_harv_EVI"),
    ("AUCcw_mean_NDVI","base_AUCcw_NDVI"),
    ("AUCcw_mean_EVI","base_AUCcw_EVI"),
]:
    if col in metrics.columns:
        metrics[f"{col}_anom"] = metrics[col] - metrics[bcol]
        with np.errstate(divide='ignore', invalid='ignore'):
            metrics[f"{col}_anom_pct"] = np.where(
                (metrics[bcol].notna()) & (metrics[bcol].abs() > 0),
                (metrics[f"{col}_anom"] / metrics[bcol]) * 100.0,
                np.nan
            )

# ---------- 10) Trends & QA ----------
trend_auc_ndvi = slope_per_adm2(metrics, "AUC_mean_NDVI")
trend_auc_evi  = slope_per_adm2(metrics, "AUC_mean_EVI")
trend_hndvi    = slope_per_adm2(metrics, "harv_mean_NDVI")
trend_hevi     = slope_per_adm2(metrics, "harv_mean_EVI")
trend = (trend_auc_ndvi.merge(trend_auc_evi, on="ADM2_PCODE", how="outer")
                        .merge(trend_hndvi,   on="ADM2_PCODE", how="outer")
                        .merge(trend_hevi,    on="ADM2_PCODE", how="outer"))

# QA: average clear fraction for union of NDVI/EVI harvest months in analysis years
if "clear_frac_mean" in data.columns:
    hw_map_nd = {r.ADM2_PCODE: _months_in_span(r.ndvi_start_month, r.ndvi_end_month)
                 for r in harv_windows.itertuples(index=False)}
    hw_map_ev = {r.ADM2_PCODE: _months_in_span(r.evi_start_month,  r.evi_end_month)
                 for r in harv_windows.itertuples(index=False)}
    union_map = {k: (hw_map_nd.get(k,set()) | hw_map_ev.get(k,set()))
                 for k in set(hw_map_nd) | set(hw_map_ev)}
    dqa = data[data["year"].isin(ANALYSIS_YEARS)].copy()
    dqa["__in_hw_union__"] = dqa.apply(lambda r: r["month"] in union_map.get(r["ADM2_PCODE"], set()), axis=1)
    qa_summary = (dqa[dqa["__in_hw_union__"]]
                  .groupby("ADM2_PCODE")["clear_frac_mean"]
                  .mean().reset_index(name="avg_clear_frac_harv_2022_2025"))
else:
    qa_summary = pd.DataFrame({"ADM2_PCODE": metrics["ADM2_PCODE"].unique(),
                               "avg_clear_frac_harv_2022_2025": np.nan})

# Final summary per ADM2 (averages for 2022–2025)
summary = (base.merge(
    metrics[metrics["year"].isin(ANALYSIS_YEARS)]
    .groupby("ADM2_PCODE")
    .agg({
        "AUC_mean_NDVI_anom":"mean",
        "AUC_mean_EVI_anom":"mean",
        "AUC_mean_NDVI_anom_pct":"mean",
        "AUC_mean_EVI_anom_pct":"mean",
        "AUCcw_mean_NDVI_anom":"mean",
        "AUCcw_mean_EVI_anom":"mean",
        "AUCcw_mean_NDVI_anom_pct":"mean",
        "AUCcw_mean_EVI_anom_pct":"mean",
        "harv_mean_NDVI_anom":"mean",
        "harv_mean_EVI_anom":"mean",
    }).rename(columns={
        "AUC_mean_NDVI_anom":"avg_AUC_NDVI_anom_2022_2025",
        "AUC_mean_EVI_anom":"avg_AUC_EVI_anom_2022_2025",
        "AUC_mean_NDVI_anom_pct":"avg_AUC_NDVI_anom_pct_2022_2025",
        "AUC_mean_EVI_anom_pct":"avg_AUC_EVI_anom_pct_2022_2025",
        "AUCcw_mean_NDVI_anom":"avg_AUCcw_NDVI_anom_2022_2025",
        "AUCcw_mean_EVI_anom":"avg_AUCcw_EVI_anom_2022_2025",
        "AUCcw_mean_NDVI_anom_pct":"avg_AUCcw_NDVI_anom_pct_2022_2025",
        "AUCcw_mean_EVI_anom_pct":"avg_AUCcw_EVI_anom_pct_2022_2025",
        "harv_mean_NDVI_anom":"avg_harv_NDVI_anom_2022_2025",
        "harv_mean_EVI_anom":"avg_harv_EVI_anom_2022_2025",
    }), on="ADM2_PCODE", how="left")
    .merge(trend, on="ADM2_PCODE", how="left")
    .merge(qa_summary, on="ADM2_PCODE", how="left"))

# ---------- 11) Save CSVs ----------
summary_path = OUTPUT_DIR / "panel_summary_by_ADM2.csv"
metrics_path = OUTPUT_DIR / "panel_yearly_metrics_long.csv"
qa_path      = OUTPUT_DIR / "panel_QA.csv"

summary.to_csv(summary_path, index=False)
metrics.to_csv(metrics_path, index=False)
qa_summary.to_csv(qa_path, index=False)

print("Saved:")
print("  ", summary_path)
print("  ", metrics_path)
print("  ", qa_path)

# ---------- 12) Aggregate & per-ADM2 plots ----------
auc_ndvi_ylim = padded_limits(auc_ndvi["AUC_mean_NDVI"], pad=0.08)
auc_evi_ylim  = padded_limits(auc_evi["AUC_mean_EVI"],   pad=0.08)

plot_metric_by_adm2(auc_ndvi, "AUC_mean_NDVI",
                    "Annual AUC (mean NDVI) by ADM2",
                    PLOT_DIR_AUC_NDVI / "overview_auc_ndvi.png",
                    ylim=auc_ndvi_ylim)
plot_metric_by_adm2(auc_evi,  "AUC_mean_EVI",
                    "Annual AUC (mean EVI) by ADM2",
                    PLOT_DIR_AUC_EVI / "overview_auc_evi.png",
                    ylim=auc_evi_ylim)
plot_metric_by_adm2(harv_ndvi, "harv_mean_NDVI",
                    "Harvest-window Mean NDVI (estimated) by ADM2",
                    PLOT_DIR_HARV_MEAN_NDVI / "overview_harvest_mean_ndvi.png",
                    ylim=(0,1))
plot_metric_by_adm2(harv_evi,  "harv_mean_EVI",
                    "Harvest-window Mean EVI (estimated) by ADM2",
                    PLOT_DIR_HARV_MEAN_EVI / "overview_harvest_mean_evi.png",
                    ylim=(0,1))

plot_per_adm2(auc_ndvi, "AUC_mean_NDVI", out_dir=PLOT_DIR_AUC_NDVI, ylim=auc_ndvi_ylim, title_prefix="AUC (NDVI) — ")
plot_per_adm2(auc_evi,  "AUC_mean_EVI",  out_dir=PLOT_DIR_AUC_EVI,  ylim=auc_evi_ylim,  title_prefix="AUC (EVI) — ")
plot_per_adm2(harv_ndvi,"harv_mean_NDVI",out_dir=PLOT_DIR_HARV_MEAN_NDVI, ylim=(0,1), title_prefix="Harvest mean NDVI (est.) — ")
plot_per_adm2(harv_evi, "harv_mean_EVI", out_dir=PLOT_DIR_HARV_MEAN_EVI,  ylim=(0,1), title_prefix="Harvest mean EVI (est.) — ")

print("Saved aggregate/per-ADM2 plots under:", PLOTS_DIR)

# ---------- 13) Rankings & compact report ----------
# Helpers
def zscore_winsor(s: pd.Series, p: float = 0.05) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2: return pd.Series(np.zeros(len(s)), index=s.index)
    lo, hi = s.quantile([p, 1-p]); s2 = s.clip(lo, hi)
    std = s2.std(ddof=0); 
    return pd.Series(np.zeros(len(s)), index=s.index) if std == 0 or np.isnan(std) else (s2 - s2.mean())/std

def pct_rank(s: pd.Series) -> pd.Series: return 100 * s.rank(pct=True, method="average")

# Fill any missing required columns
for col in [
    "base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI",
    "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
    "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
    "slope_AUC_mean_NDVI","slope_AUC_mean_EVI",
    "slope_harv_mean_NDVI","slope_harv_mean_EVI",
    "avg_clear_frac_harv_2022_2025"
]:
    if col not in summary.columns: summary[col] = np.nan

W_LEVEL = {"base_AUC_NDVI":1.0,"base_AUC_EVI":1.0,"base_harv_NDVI":0.5,"base_harv_EVI":0.5}
W_MOM   = {"avg_AUC_NDVI_anom_2022_2025":1.0,"avg_AUC_EVI_anom_2022_2025":1.0,
           "avg_harv_NDVI_anom_2022_2025":0.5,"avg_harv_EVI_anom_2022_2025":0.5,
           "slope_AUC_mean_NDVI":0.75,"slope_AUC_mean_EVI":0.75,
           "slope_harv_mean_NDVI":0.5,"slope_harv_mean_EVI":0.5}
ALPHA_LEVEL = 0.5; ALPHA_MOM = 0.5
BETA_QA = 1.0
qa_w = lambda cf: (0.5 + 0.5 * pd.to_numeric(cf, errors="coerce").fillna(1.0))

summary["LevelScore"]    = np.sum([w * zscore_winsor(summary[c]) for c, w in W_LEVEL.items()], axis=0)
summary["MomentumScore"] = np.sum([w * zscore_winsor(summary[c]) for c, w in W_MOM.items()], axis=0)
summary["CombinedScore"] = ALPHA_LEVEL*summary["LevelScore"] + ALPHA_MOM*summary["MomentumScore"]
summary["ClearWeight"]   = qa_w(summary["avg_clear_frac_harv_2022_2025"])
summary["WeightedCombined"] = summary["CombinedScore"] * (summary["ClearWeight"] ** BETA_QA)

summary["Rank_Level"]    = summary["LevelScore"].rank(ascending=False, method="min")
summary["Rank_Momentum"] = summary["MomentumScore"].rank(ascending=False, method="min")
summary["Rank_Combined"] = summary["CombinedScore"].rank(ascending=False, method="min")
summary["Rank_WeightedCombined"] = summary["WeightedCombined"].rank(ascending=False, method="min")
summary["Pct_WeightedCombined"]  = pct_rank(summary["WeightedCombined"])

leaderboard = summary.sort_values("WeightedCombined", ascending=False).reset_index(drop=True)
out_overall = OUTPUT_DIR / "adm2_rank_overall.csv"
leaderboard_cols = [
    "ADM2_PCODE","LevelScore","Rank_Level","MomentumScore","Rank_Momentum",
    "CombinedScore","Rank_Combined","ClearWeight","WeightedCombined","Rank_WeightedCombined","Pct_WeightedCombined",
    "base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI",
    "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
    "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
    "slope_AUC_mean_NDVI","slope_AUC_mean_EVI","slope_harv_mean_NDVI","slope_harv_mean_EVI",
    "avg_clear_frac_harv_2022_2025"
]
leaderboard[leaderboard_cols].to_csv(out_overall, index=False)
print("Saved overall ranking to:", out_overall)

# Per-year ranking sheet
def per_year_score(dfy):
    return (1.0*zscore_winsor(dfy["AUC_mean_NDVI"]) + 1.0*zscore_winsor(dfy["AUC_mean_EVI"]) +
            0.5*zscore_winsor(dfy["harv_mean_NDVI"]) + 0.5*zscore_winsor(dfy["harv_mean_EVI"]))
rows = []
for yr, dfy in metrics.groupby("year"):
    s = per_year_score(dfy)
    rows.append(pd.DataFrame({
        "ADM2_PCODE": dfy["ADM2_PCODE"].values,
        "year": yr,
        "YearScore": s.values,
        "YearRank": s.rank(ascending=False, method="min").values,
        "YearPct": pct_rank(s).values
    }))
yearly_rank = pd.concat(rows, ignore_index=True).sort_values(["year","YearRank"])
yearly_rank.to_csv(OUTPUT_DIR / "adm2_rank_by_year.csv", index=False)

# Compact workbook + category tables (all ADM2s)
def _cv(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return np.nan if len(s) < 3 or s.mean()==0 else float(s.std(ddof=0)/s.mean())

cv_all = (metrics.groupby("ADM2_PCODE")
          .agg(cv_auc_ndvi=("AUC_mean_NDVI", _cv),
               cv_auc_evi =("AUC_mean_EVI",  _cv))
          .reset_index())
cv_all["cv_auc_mean"] = cv_all[["cv_auc_ndvi","cv_auc_evi"]].mean(axis=1, skipna=True)

overall_leaders_full = (summary.assign(OverallRank=lambda d: d["WeightedCombined"].rank(ascending=False, method="min"))
                        .sort_values("WeightedCombined", ascending=False)
                        .loc[:, ["ADM2_PCODE","OverallRank","WeightedCombined","ClearWeight","CombinedScore","LevelScore","MomentumScore"]])
top_baseline_full = (summary.assign(BaselineRank=lambda d: d["LevelScore"].rank(ascending=False, method="min"))
                     .sort_values("LevelScore", ascending=False)
                     .loc[:, ["ADM2_PCODE","BaselineRank","LevelScore","base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI"]])
top_improvers_full = (summary.assign(ImproverRank=lambda d: d["MomentumScore"].rank(ascending=False, method="min"))
                      .sort_values("MomentumScore", ascending=False)
                      .loc[:, ["ADM2_PCODE","ImproverRank","MomentumScore",
                               "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
                               "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
                               "slope_AUC_mean_NDVI","slope_AUC_mean_EVI",
                               "slope_harv_mean_NDVI","slope_harv_mean_EVI"]])
most_stable_full = (cv_all.assign(StabilityRank=lambda d: d["cv_auc_mean"].rank(ascending=True, method="min"))
                    .sort_values(["cv_auc_mean","cv_auc_ndvi","cv_auc_evi"], ascending=[True,True,True])
                    .loc[:, ["ADM2_PCODE","StabilityRank","cv_auc_mean","cv_auc_ndvi","cv_auc_evi"]])

# Year winners (top 5 per year for narrative)
rows = []
for yr, dfy in metrics.groupby("year"):
    sc = per_year_score(dfy)
    dfy2 = dfy.assign(YearScore=sc, YearRank=sc.rank(ascending=False, method="min"))
    rows.append(dfy2[["ADM2_PCODE","year","YearScore","YearRank"]])
year_winners = (pd.concat(rows, ignore_index=True)
                .sort_values(["year","YearRank"]).groupby("year").head(5).reset_index(drop=True))

# Save workbook & CSVs
xlsx_path = OUTPUT_DIR / "adm2_category_rankings.xlsx"
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
    overall_leaders_full.to_excel(xw, sheet_name="Overall_Leaders", index=False)
    top_baseline_full.to_excel(xw, sheet_name="Top_Baseline", index=False)
    top_improvers_full.to_excel(xw, sheet_name="Top_Improvers", index=False)
    most_stable_full.to_excel(xw, sheet_name="Most_Stable", index=False)
    year_winners.to_excel(xw, sheet_name="Year_Winners", index=False)

overall_leaders_full.to_csv(OUTPUT_DIR / "adm2_overall_leaders.csv", index=False)
top_baseline_full.to_csv(OUTPUT_DIR / "adm2_top_baseline.csv", index=False)
top_improvers_full.to_csv(OUTPUT_DIR / "adm2_top_improvers.csv", index=False)
most_stable_full.to_csv(OUTPUT_DIR / "adm2_most_stable.csv", index=False)
print("Saved category outputs to:", xlsx_path)

# ---------- 14) Seasonal overlays & non-overlaid timeseries (using estimated windows) ----------
hw_map = harv_windows.set_index("ADM2_PCODE").to_dict(orient="index")
for adm2 in tqdm(sorted(data["ADM2_PCODE"].unique()), desc="Seasonal overlays + time series"):
    sub = data.loc[data["ADM2_PCODE"]==adm2, ["ADM2_PCODE","date","year","month","mean_NDVI","mean_EVI"]].copy()
    if sub.empty: continue
    # NDVI windows
    nd_s = hw_map.get(adm2, {}).get("ndvi_start_month", np.nan)
    nd_e = hw_map.get(adm2, {}).get("ndvi_end_month",   np.nan)
    seasonal_overlay(sub[["ADM2_PCODE","year","month","mean_NDVI"]].dropna(subset=["mean_NDVI"]),
                     "mean_NDVI", nd_s, nd_e, OVERLAY_NDVI_DIR / f"{adm2}_seasonal_overlay_ndvi.png")
    timeseries_with_shading(sub[["ADM2_PCODE","date","year","mean_NDVI"]].dropna(subset=["mean_NDVI"]),
                            "mean_NDVI", nd_s, nd_e, TS_NDVI_DIR / f"{adm2}_timeseries_ndvi.png")
    # EVI windows
    ev_s = hw_map.get(adm2, {}).get("evi_start_month", np.nan)
    ev_e = hw_map.get(adm2, {}).get("evi_end_month",   np.nan)
    seasonal_overlay(sub[["ADM2_PCODE","year","month","mean_EVI"]].dropna(subset=["mean_EVI"]),
                     "mean_EVI", ev_s, ev_e, OVERLAY_EVI_DIR / f"{adm2}_seasonal_overlay_evi.png")
    timeseries_with_shading(sub[["ADM2_PCODE","date","year","mean_EVI"]].dropna(subset=["mean_EVI"]),
                            "mean_EVI", ev_s, ev_e, TS_EVI_DIR / f"{adm2}_timeseries_evi.png")

print("Done. Plots in:")
print("  ", OVERLAY_NDVI_DIR)
print("  ", OVERLAY_EVI_DIR)
print("  ", TS_NDVI_DIR)
print("  ", TS_EVI_DIR)

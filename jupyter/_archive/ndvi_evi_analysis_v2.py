## Analysis
### 1) Setup & paths
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

# INPUT: folder containing files like TL0101_NDVI_EVI_monthly_admn2.csv
INPUT_DIR  = Path(r"C:\temp\timor_leste\ndvi_evi")
OUTPUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
PLOTS_DIR  = OUTPUT_DIR / "plots"

# Category subfolders for plots
PLOT_DIR_AUC_NDVI        = PLOTS_DIR / "auc_ndvi"
PLOT_DIR_AUC_EVI         = PLOTS_DIR / "auc_evi"
PLOT_DIR_HARV_MEAN_NDVI  = PLOTS_DIR / "harvest_mean_ndvi"
PLOT_DIR_HARV_MEAN_EVI   = PLOTS_DIR / "harvest_mean_evi"

# Make dirs
for d in [OUTPUT_DIR, PLOTS_DIR, PLOT_DIR_AUC_NDVI, PLOT_DIR_AUC_EVI, PLOT_DIR_HARV_MEAN_NDVI, PLOT_DIR_HARV_MEAN_EVI]:
    d.mkdir(parents=True, exist_ok=True)

# Config
BASELINE_YEARS  = [2019, 2020, 2021]
ANALYSIS_YEARS  = [2022, 2023, 2024, 2025]

# New: where the estimated windows live
HARVEST_WIN_CSV = OUTPUT_DIR / "estimated_harvest_windows.csv"
USE_CONSENSUS_FOR_BOTH = False  # if True, use consensus window for both NDVI/EVI harvest means

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 60)

### 2) Helpers
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # progress bar

def _read_one_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {fp}")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["days_in_month"] = df["date"].dt.days_in_month

    # Infer ADM2 from filename if needed
    adm2_guess = Path(fp).name.split("_")[0]
    if "ADM2_PCODE" not in df.columns:
        df["ADM2_PCODE"] = adm2_guess
    else:
        df["ADM2_PCODE"] = df["ADM2_PCODE"].fillna(adm2_guess)

    # Cast numerics
    for col in ["mean_NDVI","max_NDVI","mean_EVI","max_EVI",
                "clear_frac_mean","clear_frac_max","count_images"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_monthly_files(input_dir: Path, max_workers: int = 8) -> pd.DataFrame:
    files = glob.glob(str(input_dir / "*.csv"))
    if not files:
        raise RuntimeError(f"No CSVs found in {input_dir}")
    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_read_one_csv, fp): fp for fp in files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading CSVs"):
            fp = futures[f]
            try:
                frames.append(f.result())
            except Exception as e:
                print(f"Skipping {fp}: {e}")
    return pd.concat(frames, ignore_index=True)

# ---- harvest windows helpers ----
def _months_in_span(start_m: int, end_m: int) -> set:
    """Return a set of months {1..12} covered by a start/end month (handles wrap)."""
    if pd.isna(start_m) or pd.isna(end_m):
        return set()
    start_m, end_m = int(start_m), int(end_m)
    if 1 <= start_m <= 12 and 1 <= end_m <= 12:
        if start_m <= end_m:
            return set(range(start_m, end_m + 1))
        else:
            return set(list(range(start_m, 13)) + list(range(1, end_m + 1)))
    return set()

def _load_harvest_windows(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Harvest windows CSV not found: {csv_path}")
    hw = pd.read_csv(csv_path)
    need = {"ADM2_PCODE","ndvi_start_month","ndvi_end_month","evi_start_month","evi_end_month",
            "consensus_start_month","consensus_end_month"}
    missing = need - set(hw.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return hw

# ---- analytics helpers ----
def _ensure_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "days_in_month" not in df.columns:
            df["days_in_month"] = df["date"].dt.days_in_month
    else:
        # fallback if only year/month exist
        need = {"year","month"}
        if need.issubset(df.columns):
            if "days_in_month" not in df.columns:
                df["days_in_month"] = [
                    calendar.monthrange(int(y), int(m))[1] if pd.notna(y) and pd.notna(m) else np.nan
                    for y, m in zip(df["year"], df["month"])
                ]
        else:
            raise KeyError("Need 'date' or both 'year' & 'month' to compute days_in_month.")
    return df

def annual_auc(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    df = _ensure_calendar_cols(df)
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
    df = _ensure_calendar_cols(df)
    cols = ["ADM2_PCODE","year","days_in_month",index_col,"clear_frac_mean"]
    tmp = df.loc[:, cols].copy()

    tmp[index_col]       = pd.to_numeric(tmp[index_col], errors="coerce")
    tmp["days_in_month"] = pd.to_numeric(tmp["days_in_month"], errors="coerce")
    tmp["clear_frac_mean"] = pd.to_numeric(tmp["clear_frac_mean"], errors="coerce").fillna(0.0).clip(0,1)

    tmp["w"] = tmp[index_col] * tmp["days_in_month"] * tmp["clear_frac_mean"]
    out = (tmp.groupby(["ADM2_PCODE","year"], as_index=False)["w"].sum()
             .rename(columns={"w": f"AUCcw_{index_col}"}))
    return out



def harvest_means_estimated(df: pd.DataFrame, index_col: str,
                            hw_df: pd.DataFrame, which: str = "ndvi") -> pd.DataFrame:
    """
    Mean over the **estimated** harvest window per ADM2/year.
    which: "ndvi", "evi", or "consensus"
    Output column keeps the same name as before: harv_{index_col}
    """
    if which not in {"ndvi","evi","consensus"}:
        raise ValueError("which must be 'ndvi', 'evi', or 'consensus'")
    # Build month sets per ADM2
    hw_df = hw_df.copy()
    if which == "ndvi":
        msets = {r.ADM2_PCODE: _months_in_span(r.ndvi_start_month, r.ndvi_end_month) for r in hw_df.itertuples()}
    elif which == "evi":
        msets = {r.ADM2_PCODE: _months_in_span(r.evi_start_month, r.evi_end_month) for r in hw_df.itertuples()}
    else:
        msets = {r.ADM2_PCODE: _months_in_span(r.consensus_start_month, r.consensus_end_month) for r in hw_df.itertuples()}

    # Tag rows that fall inside each ADM2's harvest months
    df2 = df[["ADM2_PCODE","year","month",index_col]].copy()
    df2["__in_harv__"] = df2.apply(lambda r: r["month"] in msets.get(r["ADM2_PCODE"], set()), axis=1)
    sub = df2[df2["__in_harv__"]].drop(columns="__in_harv__")
    out = sub.groupby(["ADM2_PCODE","year"])[index_col].mean().reset_index(name=f"harv_{index_col}")
    return out

def slope_per_adm2(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Simple linear slope per year per ADM2."""
    rows = []
    for adm2, sub in df[["ADM2_PCODE","year",value_col]].dropna().groupby("ADM2_PCODE"):
        x = sub["year"].values.astype(float)
        y = sub[value_col].values.astype(float)
        slope = np.polyfit(x, y, 1)[0] if len(np.unique(x)) >= 2 else np.nan
        rows.append({"ADM2_PCODE": adm2, f"slope_{value_col}": slope})
    return pd.DataFrame(rows)

def plot_metric_by_adm2(df: pd.DataFrame, value_col: str, title: str, out_png: Path, ylim=None):
    """Multi-ADM2 line plot with optional fixed y-limits."""
    plt.figure()
    for adm2, sub in df.sort_values(["ADM2_PCODE","year"]).groupby("ADM2_PCODE"):
        plt.plot(sub["year"], sub[value_col], label=adm2)
    plt.title(title)
    plt.xlabel("Year"); plt.ylabel(value_col)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_per_adm2(df: pd.DataFrame, value_col: str, out_dir: Path, ylim=None, title_prefix:str=""):
    """One PNG per ADM2 with consistent y-limit, saved to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for adm2, sub in tqdm(df.groupby("ADM2_PCODE"), total=df["ADM2_PCODE"].nunique(), desc=f"Plots: {value_col}"):
        plt.figure()
        plt.plot(sub["year"], sub[value_col])
        plt.title(f"{title_prefix}{adm2}")
        plt.xlabel("Year"); plt.ylabel(value_col)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(out_dir / f"{adm2}_{value_col}.png")
        plt.close()

def padded_limits(series: pd.Series, pad: float = 0.05):
    """Compute padded [min,max] for AUC-type series; handles NaNs."""
    s = series.dropna()
    if s.empty:
        return (0, 1)
    lo, hi = s.min(), s.max()
    if lo == hi:
        span = abs(hi) if hi != 0 else 1.0
        lo, hi = hi - 0.1*span, hi + 0.1*span
    pad_span = (hi - lo) * pad
    return (lo - pad_span, hi + pad_span)

### 3) Create harvest window estimates
# ============================================================
# Estimate harvest window per ADM2 from NDVI/EVI seasonality
# ============================================================
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------- Paths / I/O ----------
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
if 'PLOTS_DIR' not in globals():
    PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Where the monthly CSVs live, if we need to reload:
if 'INPUT_DIR' in globals():
    MONTHLY_DIR = Path(INPUT_DIR)
else:
    MONTHLY_DIR = Path(r"C:\temp\timor_leste\ndvi_evi")

OUT_CSV = OUTPUT_DIR / "estimated_harvest_windows.csv"

# Optional annotated plots
MAKE_ANNOTATED_PLOTS = True
PLOT_DIR_CONS   = PLOTS_DIR / "harvest_estimate_consensus"
PLOT_DIR_NDVI   = PLOTS_DIR / "harvest_estimate_ndvi"
PLOT_DIR_EVI    = PLOTS_DIR / "harvest_estimate_evi"
for d in [PLOT_DIR_CONS, PLOT_DIR_NDVI, PLOT_DIR_EVI]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
BASELINE_YEARS = [2019, 2020, 2021]
RECENT_YEARS   = [2022, 2023, 2024, 2025]
MIN_CLEAR_FRAC = 0.20   # treat months below this clear fraction as unreliable (masked)
Y_LIMIT = (0, 1)

# ---------- Load monthly data if needed ----------
def _read_one_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {fp}")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    adm2_guess = Path(fp).name.split("_")[0]
    if "ADM2_PCODE" not in df.columns:
        df["ADM2_PCODE"] = adm2_guess
    else:
        df["ADM2_PCODE"] = df["ADM2_PCODE"].fillna(adm2_guess)
    for col in ["mean_NDVI","mean_EVI","clear_frac_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _load_monthly_if_needed():
    if 'data' in globals():
        need = {"ADM2_PCODE","year","month","mean_NDVI","mean_EVI"}
        if isinstance(data, pd.DataFrame) and need.issubset(data.columns):
            return data
    files = glob.glob(str(MONTHLY_DIR / "*.csv"))
    if not files:
        raise RuntimeError(f"No monthly CSVs found in {MONTHLY_DIR}")
    frames = []
    for fp in tqdm(files, desc="Loading monthly CSVs"):
        try:
            frames.append(_read_one_csv(fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    if not frames:
        raise RuntimeError("No valid monthly CSVs loaded.")
    return pd.concat(frames, ignore_index=True)

data = _load_monthly_if_needed()

# ---------- Utilities ----------
def _cyclic_roll(arr, shift):
    shift = shift % len(arr)
    if shift == 0: return arr.copy()
    return np.concatenate([arr[shift:], arr[:shift]])

def _smooth_cyclic(vals, window=3):
    """Centered moving average with cyclic wrap (expects len=12)."""
    assert len(vals) == 12
    x = vals.astype(float).copy()
    # simple pad with wrap:
    padded = np.r_[x[-(window//2):], x, x[:(window//2)]]
    sm = np.convolve(padded, np.ones(window)/window, mode='valid')
    return sm  # length 12

def _interp_cyclic(months, vals):
    """Interpolate NaNs on cyclic month axis."""
    m = np.array(months, dtype=float)
    v = vals.astype(float).copy()
    isnan = np.isnan(v)
    if isnan.all():
        return v
    # add wrap points (0 with Dec, 13 with Jan) to help endpoints
    m_ext = np.r_[0, m, 13]
    v_ext = np.r_[v[11], v, v[0]]
    good = ~np.isnan(v_ext)
    v_interp = np.interp(m_ext, m_ext[good], v_ext[good])
    out = v_interp[1:-1]
    return out

def _estimate_window_from_series(months, vals, clear=None):
    """
    months: 1..12
    vals: NDVI/EVI length-12 monthly mean
    clear: optional length-12 clear frac (0..1)
    Returns dict with start_month, end_month, peak_month, trough_month, range, confidence, method
    """
    months = np.array(months, dtype=int)
    x = np.array(vals, dtype=float)

    # Mask very low-clear months
    if clear is not None:
        c = np.array(clear, dtype=float)
        x = np.where((~np.isnan(c)) & (c < MIN_CLEAR_FRAC), np.nan, x)

    # Fill missing by cyclic interpolation, then smooth
    x_filled = _interp_cyclic(months, x)
    x_sm = _smooth_cyclic(x_filled, window=3)

    # Peak month (tie -> earliest)
    peak_idx = int(np.nanargmax(x_sm))
    peak_month = int(months[peak_idx])

    # Find trough within 6 months after peak (search in rolled space)
    rolled = _cyclic_roll(x_sm, peak_idx)  # starts at peak month
    # search next 1..6 (avoid the peak itself)
    search_slice = rolled[1:7]
    trough_rel = int(np.nanargmin(search_slice)) + 1
    trough_idx = (peak_idx + trough_rel) % 12
    trough_month = int(months[trough_idx])

    peak = x_sm[peak_idx]
    trough = x_sm[trough_idx]
    R = float(peak - trough)

    # thresholds
    T1 = peak - 0.20 * R  # onset when 20% down from peak
    T2 = peak - 0.60 * R  # end when 60% down from peak

    # scan forward for onset/end
    start_idx = None
    end_idx   = None
    for k in range(1, 7):  # within 6 months after peak
        idx = (peak_idx + k) % 12
        if start_idx is None and x_sm[idx] <= T1:
            start_idx = idx
        if start_idx is not None and x_sm[idx] <= T2:
            end_idx = idx
            break

    # Fallbacks if thresholds not crossed (low contrast etc.)
    method = "percent_of_range"
    if start_idx is None or end_idx is None:
        method = "steepest_decline_fallback"
        diffs = np.r_[np.diff(x_sm), x_sm[0]-x_sm[-1]]  # month-to-month cyclic
        # Look for steepest negative slope after the peak
        idxs = [(peak_idx + k) % 12 for k in range(1, 7)]
        neg_slopes = diffs[idxs]
        steep_rel = int(np.nanargmin(neg_slopes)) + 1
        center_idx = (peak_idx + steep_rel) % 12
        start_idx = start_idx or center_idx
        end_idx   = end_idx   or ((center_idx + 1) % 12)

    # Normalize ordering across year wrap
    # We'll output calendar months (1..12); if end < start, interpret as wrapping across Aug->Oct etc.
    start_month = int(months[start_idx])
    end_month   = int(months[end_idx])

    # Confidence: higher with larger R and more monotonic drop
    # monotonic score over the descent window
    seq = []
    i = start_idx
    while True:
        j = (i + 1) % 12
        seq.append(x_sm[j] - x_sm[i])
        i = j
        if i == end_idx:
            break
    monotone = float(np.mean(np.array(seq) < 0)) if seq else 0.0
    conf = np.clip((R / 0.35) * (0.5 + 0.5 * monotone), 0, 1)

    return dict(
        start_month=start_month,
        end_month=end_month,
        peak_month=peak_month,
        trough_month=trough_month,
        range_R=round(R, 3),
        confidence=round(conf, 3),
        method=method
    )

def _build_recent_climatology(df_adm2, value_col):
    """Return arrays (months 1..12) of recent-year monthly means (prefers 2022–2025)."""
    months = np.arange(1, 13, dtype=int)
    # prefer recent years if present
    yrs_present = sorted(df_adm2["year"].unique().tolist())
    yrs_recent = [y for y in RECENT_YEARS if y in yrs_present]
    if len(yrs_recent) >= 2:
        use_years = yrs_recent
    else:
        use_years = yrs_present  # fallback: all years available
    piv = (df_adm2[df_adm2["year"].isin(use_years)]
           .pivot_table(index="month", values=value_col, aggfunc="mean"))
    series = piv.reindex(months).values.ravel()
    return months, series

def _build_recent_clear(df_adm2):
    months = np.arange(1, 13, dtype=int)
    if "clear_frac_mean" not in df_adm2.columns:
        return months, np.full(12, np.nan)
    piv = (df_adm2
           .pivot_table(index="month", values="clear_frac_mean", aggfunc="mean"))
    series = piv.reindex(months).values.ravel()
    return months, series

# ---------- Main loop ----------
records = []
adm2_list = sorted(data["ADM2_PCODE"].dropna().unique().tolist())

for adm2 in tqdm(adm2_list, desc="Estimating harvest windows"):
    sub = data.loc[data["ADM2_PCODE"] == adm2,
                   ["ADM2_PCODE","year","month","mean_NDVI","mean_EVI","clear_frac_mean"]].copy()

    m_ndvi, s_ndvi = _build_recent_climatology(sub, "mean_NDVI")
    m_evi,  s_evi  = _build_recent_climatology(sub, "mean_EVI")
    m_clr,  s_clr  = _build_recent_clear(sub)

    ndvi_est = _estimate_window_from_series(m_ndvi, s_ndvi, clear=s_clr)
    evi_est  = _estimate_window_from_series(m_evi,  s_evi,  clear=s_clr)

    # Consensus = union (min start, max end). If wrap-around (e.g., start 11, end 2), normalize to May–Aug style if possible.
    # Simple union in month numbers:
    def _span_to_set(start, end):
        if start <= end:
            return set(range(start, end+1))
        else:
            return set(list(range(start, 13)) + list(range(1, end+1)))

    ndvi_set = _span_to_set(ndvi_est["start_month"], ndvi_est["end_month"])
    evi_set  = _span_to_set(evi_est["start_month"],  evi_est["end_month"])
    union    = sorted(list(ndvi_set.union(evi_set)))
    if not union:
        cons_start, cons_end = ndvi_est["start_month"], ndvi_est["end_month"]
        cons_method = "NDVI_only"
    else:
        # merge contiguous wrap-aware
        # Heuristic: prefer a window length 2–4 months around mid-year if present
        cons_start, cons_end = union[0], union[-1]
        cons_method = "union(NDVI,EVI)"

    records.append({
        "ADM2_PCODE": adm2,
        # NDVI-based
        "ndvi_start_month": ndvi_est["start_month"],
        "ndvi_end_month":   ndvi_est["end_month"],
        "ndvi_peak_month":  ndvi_est["peak_month"],
        "ndvi_trough_month":ndvi_est["trough_month"],
        "ndvi_range":       ndvi_est["range_R"],
        "ndvi_confidence":  ndvi_est["confidence"],
        "ndvi_method":      ndvi_est["method"],
        # EVI-based
        "evi_start_month":  evi_est["start_month"],
        "evi_end_month":    evi_est["end_month"],
        "evi_peak_month":   evi_est["peak_month"],
        "evi_trough_month": evi_est["trough_month"],
        "evi_range":        evi_est["range_R"],
        "evi_confidence":   evi_est["confidence"],
        "evi_method":       evi_est["method"],
        # Consensus
        "consensus_start_month": cons_start,
        "consensus_end_month":   cons_end,
        "consensus_method":      cons_method
    })

# Save CSV
harvest_df = pd.DataFrame.from_records(records)
harvest_df.to_csv(OUT_CSV, index=False)
print("Saved harvest window estimates to:", OUT_CSV)

# ---------- Optional: annotated plots ----------
def _month_labels():
    return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _plot_annotated(adm2, months, series, label, start_m, end_m, out_png):
    plt.figure()
    plt.plot(months, series, label=label)
    # Shade suggested harvest
    if start_m <= end_m:
        plt.axvspan(start_m-0.5, end_m+0.5, alpha=0.12)
    else:
        # wrap case
        plt.axvspan(0.5, end_m+0.5, alpha=0.12)
        plt.axvspan(start_m-0.5, 12.5, alpha=0.12)
    # Cosmetics
    plt.xticks(months, _month_labels())
    plt.ylim(*Y_LIMIT)
    plt.xlabel("Month"); plt.ylabel(label)
    plt.title(f"{adm2} — Estimated harvest: {start_m:02d}–{end_m:02d}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

if MAKE_ANNOTATED_PLOTS:
    for row in tqdm(harvest_df.itertuples(index=False), total=len(harvest_df), desc="Annotated plots"):
        adm2 = row.ADM2_PCODE
        sub = data.loc[data["ADM2_PCODE"] == adm2, ["year","month","mean_NDVI","mean_EVI"]]
        m_ndvi, s_ndvi = _build_recent_climatology(sub, "mean_NDVI")
        m_evi,  s_evi  = _build_recent_climatology(sub, "mean_EVI")

        # NDVI
        _plot_annotated(adm2, m_ndvi, _smooth_cyclic(_interp_cyclic(m_ndvi, s_ndvi), 3),
                        "NDVI", row.ndvi_start_month, row.ndvi_end_month,
                        PLOT_DIR_NDVI / f"{adm2}_harvest_ndvi.png")
        # EVI
        _plot_annotated(adm2, m_evi, _smooth_cyclic(_interp_cyclic(m_evi, s_evi), 3),
                        "EVI",  row.evi_start_month, row.evi_end_month,
                        PLOT_DIR_EVI / f"{adm2}_harvest_evi.png")
        # Consensus (draw NDVI curve as reference)
        _plot_annotated(adm2, m_ndvi, _smooth_cyclic(_interp_cyclic(m_ndvi, s_ndvi), 3),
                        "NDVI (consensus shading)", row.consensus_start_month, row.consensus_end_month,
                        PLOT_DIR_CONS / f"{adm2}_harvest_consensus.png")

print("Annotated plots saved in:")
print(" -", PLOT_DIR_CONS)
print(" -", PLOT_DIR_NDVI)
print(" -", PLOT_DIR_EVI)

### 3) Load data
data = load_monthly_files(INPUT_DIR, max_workers=12)
print("Rows:", len(data), " | ADM2s:", data['ADM2_PCODE'].nunique())
data.head(3)

harv_windows = _load_harvest_windows(HARVEST_WIN_CSV)

### 4) Build annual metrics (AUCs & harvest-window


auc_ndvi = annual_auc(data, "mean_NDVI")
auc_evi  = annual_auc(data, "mean_EVI")
auc_ndvi_cw = annual_auc_clearweighted(data, "mean_NDVI")
auc_evi_cw  = annual_auc_clearweighted(data, "mean_EVI")

if USE_CONSENSUS_FOR_BOTH:
    harv_ndvi = harvest_means_estimated(data, "mean_NDVI", harv_windows, which="consensus")
    harv_evi  = harvest_means_estimated(data, "mean_EVI",  harv_windows, which="consensus")
else:
    # NDVI → NDVI window; EVI → EVI window
    harv_ndvi = harvest_means_estimated(data, "mean_NDVI", harv_windows, which="ndvi")
    harv_evi  = harvest_means_estimated(data, "mean_EVI",  harv_windows, which="evi")

metrics = (auc_ndvi.merge(auc_evi, on=["ADM2_PCODE","year"], how="outer")
                    .merge(auc_ndvi_cw, on=["ADM2_PCODE","year"], how="left")
                    .merge(auc_evi_cw,  on=["ADM2_PCODE","year"], how="left")
                    .merge(harv_ndvi,   on=["ADM2_PCODE","year"], how="left")
                    .merge(harv_evi,    on=["ADM2_PCODE","year"], how="left"))

metrics.sort_values(["ADM2_PCODE","year"]).head(8)

### 5) Baseline (2019–2021) & anomalies
base = (
    metrics[metrics["year"].isin(BASELINE_YEARS)]
    .groupby("ADM2_PCODE")
    .agg({
        "AUC_mean_NDVI":"mean", "AUC_mean_EVI":"mean",
        "harv_mean_NDVI":"mean","harv_mean_EVI":"mean",
        "AUCcw_mean_NDVI":"mean","AUCcw_mean_EVI":"mean",
    })
    .rename(columns={
        "AUC_mean_NDVI":"base_AUC_NDVI",
        "AUC_mean_EVI":"base_AUC_EVI",
        "harv_mean_NDVI":"base_harv_NDVI",
        "harv_mean_EVI":"base_harv_EVI",
        "AUCcw_mean_NDVI":"base_AUCcw_NDVI",
        "AUCcw_mean_EVI":"base_AUCcw_EVI",
    })
    .reset_index()
)

metrics = metrics.merge(base, on="ADM2_PCODE", how="left")

for col, bcol in [
    ("AUC_mean_NDVI","base_AUC_NDVI"),
    ("AUC_mean_EVI","base_AUC_EVI"),
    ("harv_mean_NDVI","base_harv_NDVI"),
    ("harv_mean_EVI","base_harv_EVI"),
    ("AUCcw_mean_NDVI","base_AUCcw_NDVI"),
    ("AUCcw_mean_EVI","base_AUCcw_EVI"),
]:
    if col in metrics.columns and bcol in metrics.columns:
        metrics[f"{col}_anom"] = metrics[col] - metrics[bcol]
        with np.errstate(divide='ignore', invalid='ignore'):
            metrics[f"{col}_anom_pct"] = np.where(
                (metrics[bcol].notna()) & (metrics[bcol].abs() > 0),
                (metrics[f"{col}_anom"] / metrics[bcol]) * 100.0,
                np.nan
            )

metrics.sort_values(["ADM2_PCODE","year"]).head(8)

### 6) Trends & QA
trend_auc_ndvi = slope_per_adm2(metrics, "AUC_mean_NDVI")
trend_auc_evi  = slope_per_adm2(metrics, "AUC_mean_EVI")
trend_hndvi    = slope_per_adm2(metrics, "harv_mean_NDVI")
trend_hevi     = slope_per_adm2(metrics, "harv_mean_EVI")

trend = (trend_auc_ndvi.merge(trend_auc_evi, on="ADM2_PCODE", how="outer")
                        .merge(trend_hndvi, on="ADM2_PCODE", how="outer")
                        .merge(trend_hevi, on="ADM2_PCODE", how="outer"))

# QA: average clear fraction across the **estimated** harvest months in analysis years
if "clear_frac_mean" in data.columns:
    hw_map_ndvi = {r.ADM2_PCODE: _months_in_span(r.ndvi_start_month, r.ndvi_end_month) for r in harv_windows.itertuples()}
    hw_map_evi  = {r.ADM2_PCODE: _months_in_span(r.evi_start_month,  r.evi_end_month)  for r in harv_windows.itertuples()}
    # use union of windows to be safe for QA
    union_map = {k: (hw_map_ndvi.get(k,set()) | hw_map_evi.get(k,set())) for k in set(hw_map_ndvi)|set(hw_map_evi)}
    dqa = data[data["year"].isin(ANALYSIS_YEARS)].copy()
    dqa["__in_hw_union__"] = dqa.apply(lambda r: r["month"] in union_map.get(r["ADM2_PCODE"], set()), axis=1)
    qa_summary = (dqa[dqa["__in_hw_union__"]]
                  .groupby("ADM2_PCODE")["clear_frac_mean"]
                  .mean().reset_index(name="avg_clear_frac_harv_2022_2025"))
else:
    qa_summary = pd.DataFrame({"ADM2_PCODE": metrics["ADM2_PCODE"].unique(), "avg_clear_frac_harv_2022_2025": np.nan})

# Final summary per ADM2 (averages for 2022–2025)
summary = (
    base.merge(
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
        })
        .rename(columns={
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
        }),
        on="ADM2_PCODE", how="left"
    )
    .merge(trend, on="ADM2_PCODE", how="left")
    .merge(qa_summary, on="ADM2_PCODE", how="left")
)

summary.round(3)

### 7) Save CSVs
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

### 8) Aggregate plots (PNG)
# ---- Determine consistent y-limits ----
# AUC ranges come from data; harvest means fixed to [0,1]
auc_ndvi_ylim = padded_limits(auc_ndvi["AUC_mean_NDVI"], pad=0.08)
auc_evi_ylim  = padded_limits(auc_evi["AUC_mean_EVI"],   pad=0.08)
harv_ndvi_ylim = (0, 1)
harv_evi_ylim  = (0, 1)

# ---- Aggregate multi-ADM2 charts (one per metric) ----
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
                    ylim=harv_ndvi_ylim)

plot_metric_by_adm2(harv_evi,  "harv_mean_EVI",
                    "Harvest-window Mean EVI (estimated) by ADM2",
                    PLOT_DIR_HARV_MEAN_EVI / "overview_harvest_mean_evi.png",
                    ylim=harv_evi_ylim)

print("Saved aggregate plots to:")
print(" -", PLOT_DIR_AUC_NDVI / "overview_auc_ndvi.png")
print(" -", PLOT_DIR_AUC_EVI / "overview_auc_evi.png")
print(" -", PLOT_DIR_HARV_MEAN_NDVI / "overview_harvest_mean_ndvi.png")
print(" -", PLOT_DIR_HARV_MEAN_EVI / "overview_harvest_mean_evi.png")

### 9) Per-ADM2 plots (optional, one PNG per ADM2)
# Use the same y-limits computed above for consistency across all ADM2 plots
plot_per_adm2(auc_ndvi, "AUC_mean_NDVI",
              out_dir=PLOT_DIR_AUC_NDVI,
              ylim=auc_ndvi_ylim,
              title_prefix="AUC (NDVI) — ")

plot_per_adm2(auc_evi,  "AUC_mean_EVI",
              out_dir=PLOT_DIR_AUC_EVI,
              ylim=auc_evi_ylim,
              title_prefix="AUC (EVI) — ")

plot_per_adm2(harv_ndvi, "harv_mean_NDVI",
              out_dir=PLOT_DIR_HARV_MEAN_NDVI,
              ylim=harv_ndvi_ylim,
              title_prefix="Harvest mean NDVI (estimated) — ")

plot_per_adm2(harv_evi,  "harv_mean_EVI",
              out_dir=PLOT_DIR_HARV_MEAN_EVI,
              ylim=harv_evi_ylim,
              title_prefix="Harvest mean EVI (estimated) — ")

print("Saved per-ADM2 plots in:")
print(" -", PLOT_DIR_AUC_NDVI)
print(" -", PLOT_DIR_AUC_EVI)
print(" -", PLOT_DIR_HARV_MEAN_NDVI)
print(" -", PLOT_DIR_HARV_MEAN_EVI)

## Summary
### 1) Imports, paths, helpers
import pandas as pd
import numpy as np
from pathlib import Path

# ---- Paths ----
OUT_DIR    = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
SUMMARY_CSV = OUT_DIR / "panel_summary_by_ADM2.csv"
METRICS_CSV = OUT_DIR / "panel_yearly_metrics_long.csv"

# ---- Helper: winsorized z-score (robust to outliers/scale) ----
def zscore_winsor(s: pd.Series, p: float = 0.05) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.zeros(len(s)), index=s.index)
    lo, hi = s.quantile([p, 1 - p])
    s_clip = s.clip(lo, hi)
    std = s_clip.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s_clip - s_clip.mean()) / std

# Percentile rank (0–100)
def pct_rank(s: pd.Series) -> pd.Series:
    return 100 * s.rank(pct=True, method="average")

### 2) Load data & define score recipe
# Load your summary (ADM2-level) and yearly metrics (long)
summary = pd.read_csv(SUMMARY_CSV)
metrics = pd.read_csv(METRICS_CSV)

# If any columns are missing (older runs), create placeholders
for col in [
    "base_AUC_NDVI", "base_AUC_EVI", "base_harv_NDVI", "base_harv_EVI",
    "avg_AUC_NDVI_anom_2022_2025", "avg_AUC_EVI_anom_2022_2025",
    "avg_harv_NDVI_anom_2022_2025", "avg_harv_EVI_anom_2022_2025",
    "slope_AUC_mean_NDVI", "slope_AUC_mean_EVI",
    "slope_harv_mean_NDVI", "slope_harv_mean_EVI",
    "avg_clear_frac_harv_2022_2025"
]:
    if col not in summary.columns:
        summary[col] = np.nan

# ---- Scoring weights (tweak as desired) ----
W_LEVEL = {
    # historical level: emphasize long-season productivity (AUC), include harvest means
    "base_AUC_NDVI": 1.0,
    "base_AUC_EVI":  1.0,
    "base_harv_NDVI": 0.5,
    "base_harv_EVI":  0.5,
}
W_MOMENTUM = {
    # anomalies 2022–2025 (bigger is better)
    "avg_AUC_NDVI_anom_2022_2025": 1.0,
    "avg_AUC_EVI_anom_2022_2025":  1.0,
    "avg_harv_NDVI_anom_2022_2025": 0.5,
    "avg_harv_EVI_anom_2022_2025":  0.5,
    # slope (per year): emphasize AUC slopes, include harvest slopes
    "slope_AUC_mean_NDVI": 0.75,
    "slope_AUC_mean_EVI":  0.75,
    "slope_harv_mean_NDVI": 0.5,
    "slope_harv_mean_EVI":  0.5,
}

ALPHA_LEVEL = 0.5   # weight for Level vs Momentum in combined score
ALPHA_MOM   = 0.5

# QA weighting: downweight low clear fraction (range ~0.5..1)
# If you prefer no QA weighting, set BETA_QA=0
BETA_QA = 1.0
def qa_weight(cf):
    cf = pd.to_numeric(cf, errors="coerce").fillna(1.0)
    return 0.5 + 0.5 * cf  # 0.5 (low certainty) .. 1.0 (high clarity)

# ---- Build standardized components ----
lvl_terms = []
for col, w in W_LEVEL.items():
    z = zscore_winsor(summary[col])
    lvl_terms.append(w * z)
summary["LevelScore"] = np.sum(lvl_terms, axis=0)

mom_terms = []
for col, w in W_MOMENTUM.items():
    z = zscore_winsor(summary[col])
    mom_terms.append(w * z)
summary["MomentumScore"] = np.sum(mom_terms, axis=0)

# Combined & QA-weighted scores
summary["CombinedScore"] = ALPHA_LEVEL * summary["LevelScore"] + ALPHA_MOM * summary["MomentumScore"]
summary["ClearWeight"]   = qa_weight(summary["avg_clear_frac_harv_2022_2025"])
summary["WeightedCombined"] = summary["CombinedScore"] * (summary["ClearWeight"] ** BETA_QA)

# Ranks & percentiles (higher score = better rank)
summary["Rank_Combined"] = summary["CombinedScore"].rank(ascending=False, method="min")
summary["Rank_WeightedCombined"] = summary["WeightedCombined"].rank(ascending=False, method="min")
summary["Pct_WeightedCombined"] = pct_rank(summary["WeightedCombined"])

# Also keep separate ranks for diagnostics
summary["Rank_Level"]    = summary["LevelScore"].rank(ascending=False, method="min")
summary["Rank_Momentum"] = summary["MomentumScore"].rank(ascending=False, method="min")

# Sort for leaderboard
leaderboard = summary.sort_values("WeightedCombined", ascending=False).reset_index(drop=True)
leaderboard.head(10)

### 3) Export overall leaderboard & quick printouts
out_overall = OUT_DIR / "adm2_rank_overall.csv"
leaderboard_cols = [
    "ADM2_PCODE",
    "LevelScore","Rank_Level",
    "MomentumScore","Rank_Momentum",
    "CombinedScore","Rank_Combined",
    "ClearWeight","WeightedCombined","Rank_WeightedCombined","Pct_WeightedCombined",
    # optional context columns:
    "base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI",
    "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
    "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
    "slope_AUC_mean_NDVI","slope_AUC_mean_EVI","slope_harv_mean_NDVI","slope_harv_mean_EVI",
    "avg_clear_frac_harv_2022_2025"
]
leaderboard[leaderboard_cols].to_csv(out_overall, index=False)

print("Top 10 (QA-weighted combined):")
print(leaderboard[["ADM2_PCODE","WeightedCombined","Rank_WeightedCombined"]].head(10).to_string(index=False))

print("\nBottom 10 (QA-weighted combined):")
print(leaderboard[["ADM2_PCODE","WeightedCombined","Rank_WeightedCombined"]].tail(10).to_string(index=False))

print("\nSaved overall ranking to:", out_overall)

### 4) Per-year rankings (2019–2025) from the long “metrics” table
# Ensure needed columns exist
for col in ["AUC_mean_NDVI","AUC_mean_EVI","harv_mean_NDVI","harv_mean_EVI","year"]:
    if col not in metrics.columns:
        raise ValueError(f"Column '{col}' missing from metrics CSV.")

def per_year_score(df_year: pd.DataFrame) -> pd.Series:
    # Standardize per-year across ADM2 so each year is comparable internally
    z_auc_ndvi = zscore_winsor(df_year["AUC_mean_NDVI"])
    z_auc_evi  = zscore_winsor(df_year["AUC_mean_EVI"])
    z_h_ndvi   = zscore_winsor(df_year["harv_mean_NDVI"])
    z_h_evi    = zscore_winsor(df_year["harv_mean_EVI"])
    # Emphasize AUC, include harvest means
    return 1.0*z_auc_ndvi + 1.0*z_auc_evi + 0.5*z_h_ndvi + 0.5*z_h_evi

rows = []
for yr, dfy in metrics.groupby("year"):
    dfy = dfy.copy()
    dfy["YearScore"] = per_year_score(dfy)
    dfy["YearRank"]  = dfy["YearScore"].rank(ascending=False, method="min")
    dfy["YearPct"]   = pct_rank(dfy["YearScore"])
    rows.append(dfy[["ADM2_PCODE","year","YearScore","YearRank","YearPct"]])

yearly_rank = pd.concat(rows, ignore_index=True).sort_values(["year","YearRank"])
out_yearly = OUT_DIR / "adm2_rank_by_year.csv"
yearly_rank.to_csv(out_yearly, index=False)

print("Saved per-year rankings to:", out_yearly)
yearly_rank.head(12)

### 5) Compact report table
# ==========================================================
# Export full (all-ADM2) category rankings + updated workbook
# ==========================================================
import numpy as np
import pandas as pd
from pathlib import Path

# --- Resolve dirs & (re)load if needed ---
if 'OUTPUT_DIR' in globals():
    OUT_DIR = OUTPUT_DIR
else:
    OUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "panel_summary_by_ADM2.csv"
METRICS_CSV = OUT_DIR / "panel_yearly_metrics_long.csv"

if 'summary' not in globals() or not isinstance(summary, pd.DataFrame):
    summary = pd.read_csv(SUMMARY_CSV)
if 'metrics' not in globals() or not isinstance(metrics, pd.DataFrame):
    metrics = pd.read_csv(METRICS_CSV)

# --- Helper for CV & winsor z (small, local versions) ---
def _cv(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3 or s.mean() == 0:
        return np.nan
    return float(s.std(ddof=0) / s.mean())

def zscore_winsor(s: pd.Series, p: float = 0.05) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.zeros(len(s)), index=s.index)
    lo, hi = s.quantile([p, 1-p])
    s_clip = s.clip(lo, hi)
    std = s_clip.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s_clip - s_clip.mean()) / std

# --- Build CV table (for stability) if needed ---
cv_all = (metrics
          .groupby("ADM2_PCODE")
          .agg(cv_auc_ndvi=("AUC_mean_NDVI", _cv),
               cv_auc_evi =("AUC_mean_EVI",  _cv))
          .reset_index())
cv_all["cv_auc_mean"] = cv_all[["cv_auc_ndvi","cv_auc_evi"]].mean(axis=1, skipna=True)

# --- Full category tables (ALL ADM2s; sorted + ranks) ---
overall_leaders_full = (
    summary
    .assign(OverallRank=lambda d: d["WeightedCombined"].rank(ascending=False, method="min"))
    .sort_values(["WeightedCombined"], ascending=False)
    .loc[:, ["ADM2_PCODE","OverallRank","WeightedCombined","ClearWeight",
             "CombinedScore","LevelScore","MomentumScore"]]
)

top_baseline_full = (
    summary
    .assign(BaselineRank=lambda d: d["LevelScore"].rank(ascending=False, method="min"))
    .sort_values(["LevelScore"], ascending=False)
    .loc[:, ["ADM2_PCODE","BaselineRank","LevelScore",
             "base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI"]]
)

top_improvers_full = (
    summary
    .assign(ImproverRank=lambda d: d["MomentumScore"].rank(ascending=False, method="min"))
    .sort_values(["MomentumScore"], ascending=False)
    .loc[:, ["ADM2_PCODE","ImproverRank","MomentumScore",
             "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
             "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
             "slope_AUC_mean_NDVI","slope_AUC_mean_EVI",
             "slope_harv_mean_NDVI","slope_harv_mean_EVI"]]
)

most_stable_full = (
    cv_all
    .assign(StabilityRank=lambda d: d["cv_auc_mean"].rank(ascending=True, method="min"))
    .sort_values(["cv_auc_mean","cv_auc_ndvi","cv_auc_evi"], ascending=[True, True, True])
    .loc[:, ["ADM2_PCODE","StabilityRank","cv_auc_mean","cv_auc_ndvi","cv_auc_evi"]]
)

# --- Year winners (keep top-5 per year for narrative) ---
if 'year_winners' not in globals():
    rows = []
    for yr, dfy in metrics.groupby("year"):
        dfy = dfy.copy()
        z_auc_ndvi = zscore_winsor(dfy["AUC_mean_NDVI"])
        z_auc_evi  = zscore_winsor(dfy["AUC_mean_EVI"])
        z_h_ndvi   = zscore_winsor(dfy["harv_mean_NDVI"])
        z_h_evi    = zscore_winsor(dfy["harv_mean_EVI"])
        dfy["YearScore"] = 1.0*z_auc_ndvi + 1.0*z_auc_evi + 0.5*z_h_ndvi + 0.5*z_h_evi
        dfy["YearRank"]  = dfy["YearScore"].rank(ascending=False, method="min")
        rows.append(dfy[["ADM2_PCODE","year","YearScore","YearRank"]])
    year_winners = (pd.concat(rows, ignore_index=True)
                    .sort_values(["year","YearRank"])
                    .groupby("year").head(5).reset_index(drop=True))

# --- Compact report (same as before; keeps all ADM2s) ---
compact_cols = [
    "ADM2_PCODE",
    "Rank_WeightedCombined","WeightedCombined","ClearWeight",
    "Rank_Level","LevelScore",
    "Rank_Momentum","MomentumScore",
    "base_AUC_NDVI","base_AUC_EVI","base_harv_NDVI","base_harv_EVI",
    "avg_AUC_NDVI_anom_2022_2025","avg_AUC_EVI_anom_2022_2025",
    "avg_harv_NDVI_anom_2022_2025","avg_harv_EVI_anom_2022_2025",
    "slope_AUC_mean_NDVI","slope_AUC_mean_EVI","slope_harv_mean_NDVI","slope_harv_mean_EVI",
]
missing_compact = [c for c in compact_cols if c not in summary.columns]
if missing_compact:
    print("Note: some compact-report columns missing; output will include what's available:", missing_compact)
compact = (summary.loc[:, [c for c in compact_cols if c in summary.columns]]
           .merge(cv_all, on="ADM2_PCODE", how="left")
           .sort_values(["Rank_WeightedCombined","Rank_Level"], na_position="last")
           .reset_index(drop=True))

# --- Save workbook (overwrites existing to include full lists) ---
out_xlsx = OUT_DIR / "adm2_category_rankings.xlsx"
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
    overall_leaders_full.to_excel(xw, sheet_name="Overall_Leaders", index=False)
    top_baseline_full.to_excel(xw, sheet_name="Top_Baseline", index=False)
    top_improvers_full.to_excel(xw, sheet_name="Top_Improvers", index=False)
    most_stable_full.to_excel(xw, sheet_name="Most_Stable", index=False)
    year_winners.to_excel(xw, sheet_name="Year_Winners", index=False)
    compact.to_excel(xw, sheet_name="Compact_Report", index=False)

# --- Also save individual CSVs for convenience ---
(overall_leaders_full
 ).to_csv(OUT_DIR / "adm2_overall_leaders.csv", index=False)
(top_baseline_full
 ).to_csv(OUT_DIR / "adm2_top_baseline.csv", index=False)
(top_improvers_full
 ).to_csv(OUT_DIR / "adm2_top_improvers.csv", index=False)
(most_stable_full
 ).to_csv(OUT_DIR / "adm2_most_stable.csv", index=False)

print("Saved full-category outputs to:")
print(" -", out_xlsx)
print(" -", OUT_DIR / "adm2_overall_leaders.csv")
print(" -", OUT_DIR / "adm2_top_baseline.csv")
print(" -", OUT_DIR / "adm2_top_improvers.csv")
print(" -", OUT_DIR / "adm2_most_stable.csv")

## 3) Seasonal graphs
### Output 'seasonal_overlay_ndvi_estharvest'/'seasonal_overlay_evi_estharvest'
# ======================================================
# Recreate seasonal overlays using estimated harvest months (per ADM2)
# - NDVI overlay shades NDVI-estimated window
# - EVI  overlay shades EVI-estimated window
# ======================================================
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ---------- Paths ----------
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
if 'PLOTS_DIR' not in globals():
    PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Monthly CSV location (fallback if `data` is missing)
if 'INPUT_DIR' in globals():
    MONTHLY_DIR = Path(INPUT_DIR)
else:
    MONTHLY_DIR = Path(r"C:\temp\timor_leste\ndvi_evi")

HARVEST_CSV = OUTPUT_DIR / "estimated_harvest_windows.csv"
if not HARVEST_CSV.exists():
    raise FileNotFoundError(f"Missing {HARVEST_CSV}. Run the harvest-estimation cell first.")

# Output folders (new, so originals remain)
NDVI_DIR = PLOTS_DIR / "seasonal_overlay_ndvi_estharvest"
EVI_DIR  = PLOTS_DIR / "seasonal_overlay_evi_estharvest"
NDVI_DIR.mkdir(parents=True, exist_ok=True)
EVI_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
BASELINE_YEARS = [2019, 2020, 2021]
Y_LIMIT = (0, 1)  # consistent axis 0..1
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- Load monthly data if needed ----------
def _read_one_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {fp}")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # ADM2 from filename if absent
    adm2_guess = Path(fp).name.split("_")[0]
    if "ADM2_PCODE" not in df.columns:
        df["ADM2_PCODE"] = adm2_guess
    else:
        df["ADM2_PCODE"] = df["ADM2_PCODE"].fillna(adm2_guess)
    # Coerce key numeric fields
    for col in ["mean_NDVI","mean_EVI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _load_monthly_if_needed():
    if 'data' in globals():
        need = {"ADM2_PCODE","year","month","mean_NDVI","mean_EVI"}
        if isinstance(data, pd.DataFrame) and need.issubset(data.columns):
            return data
    files = glob.glob(str(MONTHLY_DIR / "*.csv"))
    if not files:
        raise RuntimeError(f"No monthly CSVs found in {MONTHLY_DIR}")
    frames = []
    for fp in tqdm(files, desc="Loading monthly CSVs"):
        try:
            frames.append(_read_one_csv(fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    if not frames:
        raise RuntimeError("No valid monthly CSVs loaded.")
    return pd.concat(frames, ignore_index=True)

data = _load_monthly_if_needed()

# ---------- Harvest windows ----------
harvest = pd.read_csv(HARVEST_CSV)
need_hw = {"ADM2_PCODE","ndvi_start_month","ndvi_end_month","evi_start_month","evi_end_month"}
if not need_hw.issubset(harvest.columns):
    raise ValueError(f"{HARVEST_CSV} missing columns: {need_hw - set(harvest.columns)}")

# ---------- Plot helper ----------
def _shade_window(start_m: int, end_m: int):
    """Return list of (x0, x1) spans in month-number space for shading, handling wrap-around."""
    if pd.isna(start_m) or pd.isna(end_m):
        return []
    start_m = int(start_m); end_m = int(end_m)
    if 1 <= start_m <= 12 and 1 <= end_m <= 12:
        if start_m <= end_m:
            return [(start_m - 0.5, end_m + 0.5)]
        else:
            return [(0.5, end_m + 0.5), (start_m - 0.5, 12.5)]
    return []

def _baseline_line(piv, baseline_years):
    cols = [y for y in baseline_years if y in piv.columns]
    return piv[cols].mean(axis=1) if cols else None

def _seasonal_overlay(df_adm2: pd.DataFrame, value_col: str, start_m: int, end_m: int, out_path: Path):
    # Pivot: rows=month(1..12), cols=year
    months = np.arange(1, 13, dtype=int)
    piv = (df_adm2.pivot_table(index="month", columns="year", values=value_col, aggfunc="mean")
           .reindex(index=months))

    # Plot
    plt.figure()
    # lines for each year
    for y in sorted([c for c in piv.columns if pd.notna(c)]):
        plt.plot(months, piv[y].values, label=str(int(y)))
    # baseline
    base = _baseline_line(piv, BASELINE_YEARS)
    if base is not None:
        plt.plot(months, base.values, linewidth=3, label="Baseline (2019–2021)")

    # shade harvest
    for x0, x1 in _shade_window(start_m, end_m):
        plt.axvspan(x0, x1, alpha=0.12)

    # cosmetics
    plt.xticks(months, MONTH_LABELS)
    plt.ylim(*Y_LIMIT)
    plt.xlabel("Month")
    plt.ylabel(value_col)
    adm2 = str(df_adm2["ADM2_PCODE"].iloc[0])
    metric = "NDVI" if "NDVI" in value_col else "EVI"
    if not (pd.isna(start_m) or pd.isna(end_m)):
        title_h = f" (est. harvest {MONTH_LABELS[start_m-1]}–{MONTH_LABELS[end_m-1]})"
    else:
        title_h = " (no estimate)"
    plt.title(f"{adm2} — Seasonal overlay ({metric}){title_h}")
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- Generate overlays with estimated shading ----------
adm2_list = sorted(data["ADM2_PCODE"].dropna().unique().tolist())
hw_map = harvest.set_index("ADM2_PCODE").to_dict(orient="index")

for adm2 in tqdm(adm2_list, desc="Recreating overlays with estimated harvest"):
    sub = data.loc[data["ADM2_PCODE"] == adm2, ["ADM2_PCODE","year","month","mean_NDVI","mean_EVI"]].copy()
    if sub.empty:
        continue
    # pick windows (NDVI plot uses NDVI window; EVI plot uses EVI window)
    hw = hw_map.get(adm2, {})
    ndvi_s = hw.get("ndvi_start_month", np.nan); ndvi_e = hw.get("ndvi_end_month", np.nan)
    evi_s  = hw.get("evi_start_month",  np.nan); evi_e  = hw.get("evi_end_month",  np.nan)

    # NDVI overlay
    _seasonal_overlay(
        sub[["ADM2_PCODE","year","month","mean_NDVI"]].dropna(subset=["mean_NDVI"]),
        "mean_NDVI",
        ndvi_s, ndvi_e,
        NDVI_DIR / f"{adm2}_seasonal_overlay_ndvi.png"
    )
    # EVI overlay
    _seasonal_overlay(
        sub[["ADM2_PCODE","year","month","mean_EVI"]].dropna(subset=["mean_EVI"]),
        "mean_EVI",
        evi_s, evi_e,
        EVI_DIR / f"{adm2}_seasonal_overlay_evi.png"
    )

print("Done. Saved updated overlays to:")
print(" -", NDVI_DIR)
print(" -", EVI_DIR)

### Output 'timeseries_ndvi_estharvest'/'timeseries_evi_estharvest'
# ======================================================
# Non-overlaid monthly time series with per-year harvest shading
# ======================================================
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from pathlib import Path
from tqdm import tqdm
import calendar
from datetime import datetime

# ---------- Paths ----------
if 'OUTPUT_DIR' not in globals():
    OUTPUT_DIR = Path(r"C:\temp\timor_leste\ndvi_evi_outputs")
if 'PLOTS_DIR' not in globals():
    PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NDVI_TS_DIR = PLOTS_DIR / "timeseries_ndvi_estharvest"
EVI_TS_DIR  = PLOTS_DIR / "timeseries_evi_estharvest"
NDVI_TS_DIR.mkdir(parents=True, exist_ok=True)
EVI_TS_DIR.mkdir(parents=True, exist_ok=True)

# Monthly CSV location if `data` is missing
if 'INPUT_DIR' in globals():
    MONTHLY_DIR = Path(INPUT_DIR)
else:
    MONTHLY_DIR = Path(r"C:\temp\timor_leste\ndvi_evi")

HARVEST_CSV = OUTPUT_DIR / "estimated_harvest_windows.csv"
if not HARVEST_CSV.exists():
    raise FileNotFoundError(f"Missing {HARVEST_CSV}. Run the harvest-estimation cell first.")
harvest_df = pd.read_csv(HARVEST_CSV)

# ---------- Load monthly data if needed ----------
def _read_one_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {fp}")
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # ADM2 from filename if absent
    adm2_guess = Path(fp).name.split("_")[0]
    if "ADM2_PCODE" not in df.columns:
        df["ADM2_PCODE"] = adm2_guess
    else:
        df["ADM2_PCODE"] = df["ADM2_PCODE"].fillna(adm2_guess)
    # Coerce useful numeric fields
    for col in ["mean_NDVI","mean_EVI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _load_monthly_if_needed():
    if 'data' in globals():
        need = {"ADM2_PCODE","date","year","month","mean_NDVI","mean_EVI"}
        if isinstance(data, pd.DataFrame) and need.issubset(data.columns):
            return data
    files = glob.glob(str(MONTHLY_DIR / "*.csv"))
    if not files:
        raise RuntimeError(f"No monthly CSVs found in {MONTHLY_DIR}")
    frames = []
    for fp in tqdm(files, desc="Loading monthly CSVs"):
        try:
            frames.append(_read_one_csv(fp))
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    if not frames:
        raise RuntimeError("No valid monthly CSVs loaded.")
    return pd.concat(frames, ignore_index=True)

data = _load_monthly_if_needed()

# Safety check
need_cols = {"ADM2_PCODE","date","year","month","mean_NDVI","mean_EVI"}
missing = need_cols - set(data.columns)
if missing:
    raise ValueError(f"Monthly data missing columns: {missing}.")

# ---------- Helpers ----------
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _eom(year: int, month: int) -> datetime:
    """End-of-month date."""
    day = calendar.monthrange(year, month)[1]
    return datetime(year, month, day)

def _shades_for_year(y: int, start_m: int, end_m: int):
    """
    Return list of (start_date, end_date) spans within calendar year y
    to shade, given a start/end month (handles wrap).
    """
    if pd.isna(start_m) or pd.isna(end_m):
        return []
    start_m = int(start_m); end_m = int(end_m)
    spans = []
    if 1 <= start_m <= 12 and 1 <= end_m <= 12:
        if start_m <= end_m:
            spans.append((datetime(y, start_m, 1), _eom(y, end_m)))
        else:
            # wrap: shade Jan..end_m and start_m..Dec inside the SAME calendar year
            spans.append((datetime(y, 1, 1), _eom(y, end_m)))
            spans.append((datetime(y, start_m, 1), datetime(y, 12, 31)))
    return spans

def _plot_timeseries_with_shading(df_adm2: pd.DataFrame,
                                  value_col: str,
                                  start_m: int, end_m: int,
                                  out_path: Path):
    dfp = df_adm2.sort_values("date")
    # Plot
    fig, ax = plt.subplots()
    ax.plot(dfp["date"], dfp[value_col], marker="o", linewidth=1.5, label=value_col)

    # Per-year shading using estimated window months
    for y in sorted(dfp["year"].unique()):
        for d0, d1 in _shades_for_year(y, start_m, end_m):
            # limit to plotting range
            left  = max(d0, dfp["date"].min().to_pydatetime())
            right = min(d1, dfp["date"].max().to_pydatetime())
            if left <= right:
                ax.axvspan(left, right, alpha=0.12)

    # Cosmetics
    ax.set_ylim(0, 1)  # consistent scale
    ax.set_ylabel(value_col)
    ax.set_xlabel("Date")
    adm2 = str(dfp["ADM2_PCODE"].iloc[0])
    if not (pd.isna(start_m) or pd.isna(end_m)):
        title_h = f" (est. harvest {MONTH_NAMES[start_m-1]}–{MONTH_NAMES[end_m-1]})"
    else:
        title_h = " (no harvest estimate)"
    ax.set_title(f"{adm2} — Monthly time series {title_h}")

    # Ticks/formatters
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ---------- Build maps for quick lookup ----------
hw = harvest_df.set_index("ADM2_PCODE").to_dict(orient="index")

# ---------- Generate plots ----------
adm2_list = sorted(data["ADM2_PCODE"].dropna().unique().tolist())

for adm2 in tqdm(adm2_list, desc="Time series with harvest shading"):
    sub = data.loc[data["ADM2_PCODE"] == adm2, ["ADM2_PCODE","date","year","month","mean_NDVI","mean_EVI"]].dropna(subset=["date"]).copy()
    if sub.empty:
        continue
    # Ensure datetime dtype
    sub["date"] = pd.to_datetime(sub["date"])

    # NDVI window
    ndvi_s = hw.get(adm2, {}).get("ndvi_start_month", np.nan)
    ndvi_e = hw.get(adm2, {}).get("ndvi_end_month",   np.nan)
    _plot_timeseries_with_shading(sub[["ADM2_PCODE","date","year","mean_NDVI"]].dropna(subset=["mean_NDVI"]),
                                  "mean_NDVI", ndvi_s, ndvi_e,
                                  NDVI_TS_DIR / f"{adm2}_timeseries_ndvi.png")

    # EVI window
    evi_s  = hw.get(adm2, {}).get("evi_start_month",  np.nan)
    evi_e  = hw.get(adm2, {}).get("evi_end_month",    np.nan)
    _plot_timeseries_with_shading(sub[["ADM2_PCODE","date","year","mean_EVI"]].dropna(subset=["mean_EVI"]),
                                  "mean_EVI",  evi_s, evi_e,
                                  EVI_TS_DIR  / f"{adm2}_timeseries_evi.png")

print("Saved time-series plots to:")
print(" -", NDVI_TS_DIR)
print(" -", EVI_TS_DIR)

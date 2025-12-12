#AI acknowledgement: This file contains AI generated code used for parsing metadata .pkl's
from __future__ import annotations

import os
import json
import glob
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

FILE_GLOB_PATTERN: str = "human_metadata_*.pkl"  # e.g., "species_*_*_metadata.pkl"

# If provided, ONLY these fields will have using=True. All others become using=False.
# If None, all fields are marked using=False (but still fully preprocessed).

# Columns that are NOT metadata and should be ignored entirely.
IGNORE_UTILITY_COLUMNS: Iterable[str] = ("_source_file",)

# Optional: where to save artifacts (set to None to skip saving)
SAVE_DIR: Optional[str] = None  # e.g., os.path.join(DATA_DIR, "preproc")
SPECS_JSON_NAME: str = "metadata_field_specs.json"
VOCAB_JSON_NAME: str = "metadata_vocab.json"



# =========================
# 2) IO Utilities
# =========================

def list_metadata_files(data_dir: str, pattern: str, *, verbose: bool = False) -> List[str]:
    """Return a sorted list of metadata pickle file paths matching the pattern inside data_dir."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if verbose:
        print(f"[list] Searching in: {data_dir!r} pattern: {pattern!r}")
        print(f"[list] Found {len(paths)} file(s).")
    if not paths:
        raise FileNotFoundError(f"No files found under {data_dir!r} with pattern {pattern!r}.")
    return paths


def _safe_read_pickle(path: str, *, verbose: bool = False) -> Any:
    """
    Read a pickle file safely.
    Accepts pandas pickles (DataFrame/Series) or python-native objects (list/dict of records).
    """
    try:
        obj = pd.read_pickle(path)
        if verbose:
            print(f"[read] Loaded via pandas: {os.path.basename(path)} ({type(obj).__name__})")
        return obj
    except Exception:
        if verbose:
            print(f"[read] pandas failed, using pickle: {os.path.basename(path)}")
        with open(path, "rb") as f:
            return pickle.load(f)


def load_metadata_as_dataframe(filepaths: List[str], *, verbose: bool = False) -> pd.DataFrame:
    """
    Load a set of pickle metadata files into a single DataFrame.
    Adds a column '_source_file' indicating the origin of each record.
    """
    frames: List[pd.DataFrame] = []
    for p in filepaths:
        obj = _safe_read_pickle(p, verbose=verbose)

        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, pd.Series):
            df = obj.to_frame().T
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        elif isinstance(obj, dict):
            try:
                df = pd.DataFrame([obj])  # single-record interpretation
            except Exception:
                df = pd.DataFrame(obj)    # dict-of-lists interpretation
        else:
            raise ValueError(f"Unsupported pickle content type {type(obj)} in '{p}'.")

        df["_source_file"] = os.path.basename(p)
        frames.append(df)

    if not frames:
        raise RuntimeError("No metadata could be loaded into a DataFrame.")

    out = pd.concat(frames, axis=0, ignore_index=True, join="outer")
    if verbose:
        print(f"[load] Concatenated {len(frames)} frame(s): rows={len(out)}, cols={len(out.columns)}")
    return out




# =========================
# 3) Normalization & Type Helpers
# =========================

def normalize_value(v: Any) -> Any:
    """
    Normalize categorical values so they are hashable & consistent.
    - Convert NaN-like to np.nan
    - Convert lists/tuples/sets to a tuple of strings
    - Strip whitespace for strings
    - Leave numbers as-is (np.nan already handled)
    - Fallback: stripped string or np.nan if empty
    """
    if pd.isna(v):
        return np.nan

    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else np.nan

    if isinstance(v, (list, tuple, set)):
        return tuple(str(x).strip() for x in v)

    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
        return v

    s = str(v).strip()
    return s if s != "" else np.nan


def normalize_column(series: pd.Series) -> pd.Series:
    """Apply normalize_value element-wise."""
    return series.map(normalize_value)


#TODO: Cant just be typeof(bool) or something? isinstance? 
def _is_boolean_like(s: pd.Series) -> bool:
    """Return True if the non-null set is a subset of {0,1,True,False}."""
    if s.empty:
        return False
    non_null = s.dropna()
    allowed = {0, 1, True, False}
    try:
        return non_null.map(lambda x: x in allowed).all()
    except Exception:
        return False


def detect_metadata_kind(series: pd.Series) -> str:
    """
    Heuristically classify a column after normalization:
      - 'boolean'            : values are in {0,1,True,False}
      - 'numeric'            : numeric dtype and not boolean-like
      - 'datetime'           : pandas datetime dtype
      - 'categorical_multi'  : majority are tuples (multi-valued categories)
      - 'categorical'        : everything else (strings/objects/categories)
      - 'empty'              : all null
    """
    s = normalize_column(series)
    s_nz = s.dropna()

    if s_nz.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(s_nz):
        return "datetime"
    if _is_boolean_like(s_nz):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s_nz):
        return "numeric"
    # Multi-valued check
    try:
        frac_tuple = float((s_nz.map(lambda x: isinstance(x, tuple))).mean())
        if frac_tuple > 0.5:
            return "categorical_multi"
    except Exception:
        pass
    return "categorical"






# =========================
# 4) Vocab & Specs Builders
# =========================

def compute_non_null_fraction(series: pd.Series) -> float:
    """Fraction of entries that are not null/NaN."""
    denom = len(series)
    if denom == 0:
        return 0.0
    return float(series.notna().mean())


def compute_field_vocab(
    series: pd.Series,
    dropna: bool = True,
    sort_values: bool = True,
) -> Dict[Any, int]:
    """Build a value->index mapping for a normalized categorical/boolean series."""
    s = normalize_column(series)
    if dropna:
        s = s.dropna()

    uniques = pd.unique(s)
    if sort_values:
        uniques = sorted(uniques, key=lambda x: str(x))
    return {val: idx for idx, val in enumerate(uniques)}


def build_vocab_and_specs(
    df: pd.DataFrame,
    include_fields: Optional[Iterable[str]] = None,
    ignore_utility_columns: Iterable[str] = ("_source_file",),
    *,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[Any, int]], Dict[str, Dict[str, Any]]]:
    """
    Build:
      vocab_mapping (dict[str, dict[Any,int]]):
        - For boolean/categorical fields, FULL vocab (value->idx) of non-null normalized values.
        - For numeric/datetime/empty, vocab is {} (we don't one-hot continuous/time by default).
      specs_dict (dict[str, dict]):
        {
          "cardinality": int (unique non-null),
          "using": bool (True iff field in include_fields),
          "non_null_fraction": float,
          "non_null_count": int,
          "null_count": int,
          "total_count": int,
          "kind": one of {"boolean","categorical","categorical_multi","numeric","datetime","empty"},
          # Optional summaries:
          "numeric_summary": {"min":..., "max":..., "mean":..., "std":..., "p50":...}  # numeric only
        }
    Notes:
      - All non-utility columns are preprocessed; 'using' is JUST a switch, not a filter.
    """
    vocab: Dict[str, Dict[Any, int]] = {}
    specs: Dict[str, Dict[str, Any]] = {}

    utility_set = set(ignore_utility_columns)
    fields = [c for c in df.columns if c not in utility_set]

    if verbose:
        print(f"[build] Processing {len(fields)} field(s) (ignoring: {sorted(utility_set)})")
        if include_fields is None:
            print("[build] No include list provided -> all fields will be marked using=False.")
        else:
            inc = set(include_fields)
            print(f"[build] Include list provided ({len(inc)} field(s)) -> using=True iff in include list.")

    for col in fields:
        series = df[col]
        knd = detect_metadata_kind(series)
        total = int(series.shape[0])
        non_null_count = int(series.notna().sum())
        null_count = total - non_null_count
        nn_frac = round(non_null_count / total, 6) if total > 0 else 0.0

        # Compute unique non-null values AFTER normalization (for robust cardinality).
        s_norm = normalize_column(series).dropna()
        unique_non_null = int(pd.unique(s_norm).shape[0])

        # Determine 'using' strictly from include_fields
        use_flag = (include_fields is not None) and (col in set(include_fields))

        # Build vocab only for boolean/categorical flavors
        if knd in {"boolean", "categorical", "categorical_multi"}:
            vmap = compute_field_vocab(series, dropna=True, sort_values=True)
        else:
            vmap = {}  # numeric/datetime/empty: not one-hot by default

        # Numeric summary if applicable
        numeric_summary = None
        if knd == "numeric" and non_null_count > 0:
            try:
                s_num = pd.to_numeric(s_norm, errors="coerce").dropna()
                if not s_num.empty:
                    numeric_summary = {
                        "min": float(np.nanmin(s_num)),
                        "max": float(np.nanmax(s_num)),
                        "mean": float(np.nanmean(s_num)),
                        "std": float(np.nanstd(s_num, ddof=0)),
                        "p50": float(np.nanpercentile(s_num, 50)),
                    }
            except Exception:
                numeric_summary = None

        # Specs entry
        specs[col] = {
            "cardinality": int(unique_non_null),
            "using": bool(use_flag),
            "non_null_fraction": float(nn_frac),
            "non_null_count": int(non_null_count),
            "null_count": int(null_count),
            "total_count": int(total),
            "kind": knd,
        }
        if numeric_summary is not None:
            specs[col]["numeric_summary"] = numeric_summary

        vocab[col] = vmap

        if verbose:
            print(
                f"[field] {col!r}: kind={knd:>16} | using={use_flag!s:>5} | "
                f"card={unique_non_null:>7} | nn_frac={nn_frac:.6f} | "
                f"vocab_sz={len(vmap):>7}"
            )

    return vocab, specs




# =========================
# 5) Serialization
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any, *, verbose: bool = False) -> None:
    """Save Python object as JSON with UTF-8 encoding."""
    if verbose:
        print(f"[save] Writing: {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def specs_to_jsonable(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Specs is already dict-of-dicts; ensure primitive types & rounding for stability."""
    out: Dict[str, Dict[str, Any]] = {}
    for field, d in specs.items():
        rec: Dict[str, Any] = {
            "cardinality": int(d.get("cardinality", 0)),
            "using": bool(d.get("using", False)),
            "non_null_fraction": round(float(d.get("non_null_fraction", 0.0)), 6),
            "non_null_count": int(d.get("non_null_count", 0)),
            "null_count": int(d.get("null_count", 0)),
            "total_count": int(d.get("total_count", 0)),
            "kind": str(d.get("kind", "unknown")),
        }
        if "numeric_summary" in d and isinstance(d["numeric_summary"], dict):
            ns = d["numeric_summary"]
            rec["numeric_summary"] = {
                k: float(ns[k]) for k in ["min", "max", "mean", "std", "p50"] if k in ns
            }
        out[field] = rec
    return out


def vocab_to_jsonable(vocab: Dict[str, Dict[Any, int]]) -> Dict[str, Dict[str, int]]:
    """
    Convert vocab values (which might be non-JSON types like tuples) to strings for JSON safety.
    Keys become strings; indices remain ints.
    """
    out: Dict[str, Dict[str, int]] = {}
    for field, mapping in vocab.items():
        out[field] = {str(k): int(v) for k, v in mapping.items()}
    return out





# =========================
# 6) Pipeline Entrypoint (Notebook-friendly)
# =========================

def run_metadata_preprocessing(
    data_dir: str,
    file_glob_pattern: str,
    include_fields: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    specs_json_name: str = "metadata_field_specs.json",
    vocab_json_name: str = "metadata_vocab.json",
    ignore_utility_columns: Iterable[str] = IGNORE_UTILITY_COLUMNS,
    *,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[Any, int]], Dict[str, Dict[str, Any]]]:
    """
    Full pipeline (verbose prints guide each step):
      1) List files
      2) Load to DataFrame
      3) Build vocab + specs (preprocess ALL non-utility fields; 'using' strictly from include list)
      4) (Optional) Save JSON artifacts
    Returns:
      vocab_mapping, specs_dict
    """
    if verbose:
        print("=== METADATA PREPROCESSING START ===")
        print(f"[cfg] data_dir={data_dir!r}")
        print(f"[cfg] file_glob_pattern={file_glob_pattern!r}")
        print(f"[cfg] include_fields={'None' if include_fields is None else list(include_fields)}")
        print(f"[cfg] save_dir={save_dir!r}")
        print(f"[cfg] ignore_utility_columns={tuple(ignore_utility_columns)}")

    filepaths = list_metadata_files(data_dir, file_glob_pattern, verbose=verbose)
    df = load_metadata_as_dataframe(filepaths, verbose=verbose)

    vocab, specs = build_vocab_and_specs(
        df=df,
        include_fields=include_fields,
        ignore_utility_columns=ignore_utility_columns,
        verbose=verbose,
    )

    if save_dir:
        ensure_dir(save_dir)
        specs_json = specs_to_jsonable(specs)
        vocab_json = vocab_to_jsonable(vocab)
        save_json(os.path.join(save_dir, specs_json_name), specs_json, verbose=verbose)
        save_json(os.path.join(save_dir, vocab_json_name), vocab_json, verbose=verbose)
        if verbose:
            print(f"[done] Saved specs -> {os.path.join(save_dir, specs_json_name)}")
            print(f"[done] Saved vocab -> {os.path.join(save_dir, vocab_json_name)}")

    if verbose:
        print("=== METADATA PREPROCESSING COMPLETE ===")

    return vocab, specs



def main(raw_args=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Metadata preprocessing to build specs (dict-of-dicts) and vocab mappings."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing metadata pickle files.")
    parser.add_argument("--pattern", default=FILE_GLOB_PATTERN,
                        help=f"Glob pattern within data-dir (default: {FILE_GLOB_PATTERN!r}).")
    # include list controls 'using' exactly; all others become using=False
    parser.add_argument("--include", action="append", default=None,
                        help="Field to mark using=True (repeatable). If omitted, all fields using=False.")
    parser.add_argument("--include-file", type=str, default=None,
                        help="Optional path to a text file with one field name per line to include.")
    # outputs
    parser.add_argument("--save-dir", default=None,
                        help="Directory to write JSON artifacts. If omitted, defaults to <data-dir>/preproc.")
    parser.add_argument("--specs-json-name", default=SPECS_JSON_NAME,
                        help=f"Specs filename (default: {SPECS_JSON_NAME!r}).")
    parser.add_argument("--vocab-json-name", default=VOCAB_JSON_NAME,
                        help=f"Vocab filename (default: {VOCAB_JSON_NAME!r}).")
    # misc
    parser.add_argument("--ignore-column", action="append",
                        default=list(IGNORE_UTILITY_COLUMNS),
                        help=f"Columns to treat as utility/provenance (repeatable). "
                             f"Default: {tuple(IGNORE_UTILITY_COLUMNS)}")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except writing JSON files.")

    args = parser.parse_args(raw_args)
    print(f"Running metadata_preprocessor with args: {vars(args)}")

    # build include_fields set from --include and/or --include-file
    include_fields = None
    if args.include or args.include_file:
        include_set = set(args.include or [])
        if args.include_file:
            with open(args.include_file, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name and not name.startswith("#"):
                        include_set.add(name)
        include_fields = sorted(include_set) if include_set else None  # None => all using=False

    # decide save directory
    save_dir = None
    if not args.dry_run:
        save_dir = args.save_dir or SAVE_DIR or str(Path(args.data_dir) / "preproc")

    vocab, specs = run_metadata_preprocessing(
        data_dir=args.data_dir,
        file_glob_pattern=args.pattern,
        include_fields=include_fields,
        save_dir=save_dir,
        specs_json_name=args.specs_json_name,
        vocab_json_name=args.vocab_json_name,
        ignore_utility_columns=args.ignore_column,
        verbose=args.verbose,
    )

    # Summary to stdout (useful on HPC logs)
    total_fields = len(specs)
    using_fields = sum(1 for d in specs.values() if d.get("using", False))
    if args.verbose:
        print(f"[summary] fields={total_fields}, using={using_fields}, "
              f"with_vocab={sum(1 for v in vocab.values() if len(v) > 0)}")
        if args.dry_run:
            print("[summary] dry-run enabled; no files written.")



if __name__ == "__main__":
   main()
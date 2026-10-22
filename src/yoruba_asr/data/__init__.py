"""Dataset loaders. Every loader returns a DataFrame with at least:
``path, sentence, speaker, corpus, split, gender, age``."""

STANDARD_COLUMNS = ["path", "sentence", "speaker", "corpus", "split", "gender", "age"]


def ensure_columns(df):
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in ("speaker",) else "unknown"
    df["speaker"] = df["speaker"].fillna("unknown").astype(str).replace("", "unknown")
    for col in ("gender", "age"):
        df[col] = df[col].fillna("").astype(str)
    return df

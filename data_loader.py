from io import BytesIO, StringIO
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
SEPARATORS = (None, ",", ";", "\t", "|")
EXCEL_EXTENSIONS = (".xlsx", ".xls")
CSV_EXTENSIONS = (".csv", ".txt", ".tsv", "")

ID_REGEX_PATTERNS = (
    r"^(id|cod|codigo|código|protocolo|ticket|chave|key|guid|uuid|cpf|cnpj|matricula|matrícula)(_.*)?$",
    r".*(_id|_cod|_codigo|_código|_protocolo|_ticket|_chave|_key|_uuid)$",
)

DATE_KEYWORDS = (
    "data", "date", "dt_", "time", "timestamp", "criado", "created",
    "abertura", "fechamento", "resolvido", "solved", "recieved", "received",
    "updated", "modificado", "prazo", "vencimento", "nascimento",
)

TEXT_KEYWORDS = (
    "text", "desc", "coment", "comment", "feedback", "obs", "nota",
    "mensagem", "justificativa",
)


def _rewind_stream(source: Any) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _detect_file_extension(file_source: Any) -> str:
    name = getattr(file_source, "name", file_source)
    return Path(str(name)).suffix.lower() if isinstance(name, (str, Path)) else ""


def _read_csv_with_fallback(file_source: Any) -> pd.DataFrame:
    last_error: Optional[Exception] = None

    for encoding in ENCODINGS:
        for sep in SEPARATORS:
            try:
                _rewind_stream(file_source)
                read_kwargs = {"encoding": encoding}
                if sep is None:
                    read_kwargs.update({"sep": None, "engine": "python"})
                else:
                    read_kwargs.update({"sep": sep})

                df = pd.read_csv(file_source, **read_kwargs)
                if not df.empty and df.shape[1] > 1:
                    return df
            except UnicodeDecodeError as err:
                last_error = err
                break
            except Exception as err:
                last_error = err
                continue

    try:
        _rewind_stream(file_source)
        return pd.read_csv(file_source, encoding="latin-1", sep=",")
    except Exception as final_err:
        raise ValueError(
            f"Não foi possível decodificar o arquivo delimitado: {last_error or final_err}"
        )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    df_clean = df_clean.dropna(how="all")

    valid_cols = [c for c in df_clean.columns if c and not c.startswith("Unnamed:")]
    if valid_cols:
        df_clean = df_clean[valid_cols]
    df_clean = df_clean.dropna(axis=1, how="all")

    string_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].apply(
            lambda val: val.strip() if isinstance(val, str) else val
        )

    return df_clean


def _is_temporal_column(series: pd.Series, col_lower: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(keyword in col_lower for keyword in DATE_KEYWORDS):
        non_null = series.dropna()
        if not non_null.empty:
            try:
                sample = non_null.head(20)
                parsed = pd.to_datetime(sample, errors="coerce")
                return bool(parsed.notnull().sum() >= len(sample) * 0.7)
            except Exception:
                pass
    return False


def _is_identifier_column(series: pd.Series, col_lower: str, total_rows: int, unique_ratio: float) -> bool:
    if any(re.match(pattern, col_lower) for pattern in ID_REGEX_PATTERNS):
        return True
    if total_rows >= 10 and unique_ratio >= 0.98:
        non_null = series.dropna()
        avg_len = non_null.astype(str).str.len().mean() if not non_null.empty else 0
        return bool(pd.api.types.is_numeric_dtype(series) or (avg_len < 35))
    return False


def _classify_column(series: pd.Series, col_name: str, total_rows: int, n_unique: int, unique_ratio: float) -> str:
    col_lower = col_name.strip().lower()

    if _is_temporal_column(series, col_lower):
        return "temporal"

    if _is_identifier_column(series, col_lower, total_rows, unique_ratio):
        return "identifiers"

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 2 and unique_ratio < 0.1:
            return "categorical"
        return "numeric_metrics"

    non_null = series.dropna()
    avg_len = float(non_null.astype(str).str.len().mean()) if not non_null.empty else 0.0
    has_text_keyword = any(k in col_lower for k in TEXT_KEYWORDS)

    if avg_len >= 35 or (has_text_keyword and avg_len >= 15) or (n_unique > 50 and unique_ratio > 0.4):
        return "text"

    return "categorical"


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    categories = ("categorical", "numeric_metrics", "text", "identifiers", "temporal")
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "classifications": {cat: [] for cat in categories},
            "details": {},
        }

    total_rows = len(df)
    classifications: Dict[str, List[str]] = {cat: [] for cat in categories}
    details: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        null_count = int(series.isnull().sum())
        null_pct = round((null_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        n_unique = int(non_null.nunique())
        unique_ratio = (n_unique / len(non_null)) if len(non_null) > 0 else 0.0

        inferred_type = _classify_column(series, str(col), total_rows, n_unique, unique_ratio)
        classifications[inferred_type].append(str(col))

        details[str(col)] = {
            "inferred_type": inferred_type,
            "original_dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": n_unique,
            "unique_ratio": round(unique_ratio, 4),
        }

    classifications["categorical"].sort(
        key=lambda c: (details[c]["null_pct"], details[c]["unique_count"])
    )

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "classifications": classifications,
        "details": details,
    }


def load_and_validate_data(
    file_source: Union[str, Path, BytesIO, StringIO, Any],
    required_columns: Optional[Union[List[str], Dict[str, Any]]] = None,
) -> Tuple[Optional[pd.DataFrame], bool, str]:
    if file_source is None:
        return None, False, "Nenhum arquivo fornecido para carregamento."

    ext = _detect_file_extension(file_source)

    try:
        if ext in EXCEL_EXTENSIONS:
            _rewind_stream(file_source)
            df = pd.read_excel(file_source)
        elif ext in CSV_EXTENSIONS:
            df = _read_csv_with_fallback(file_source)
        else:
            return None, False, f"Formato '{ext}' não suportado. Utilize arquivos .csv, .tsv ou .xlsx."
    except Exception as exc:
        return None, False, f"Falha ao ler o arquivo: {exc}"

    if df is None or df.empty:
        return None, False, "O arquivo carregado está vazio ou não contém registros legíveis."

    df_clean = _clean_dataframe(df)
    if df_clean.empty:
        return None, False, "O arquivo não contém linhas preenchidas após a higienização."

    schema = detect_schema(df_clean)
    classes = schema["classifications"]

    type_labels = {
        "categorical": "categórica(s)",
        "numeric_metrics": "métrica(s) numérica(s)",
        "text": "textual(is)",
        "identifiers": "identificador(es)",
        "temporal": "temporal(is)",
    }
    summary_parts = [
        f"{len(classes[k])} {label}"
        for k, label in type_labels.items()
        if classes.get(k)
    ]
    summary_text = ", ".join(summary_parts) if summary_parts else f"{schema['total_columns']} colunas"
    msg = f"Arquivo carregado com sucesso! {schema['total_rows']:,} linhas e {schema['total_columns']} colunas identificadas ({summary_text})."

    return df_clean, True, msg


def load_and_profile_data(
    file_source: Union[str, Path, BytesIO, StringIO, Any],
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], bool, str]:
    df, success, msg = load_and_validate_data(file_source)
    if not success or df is None:
        return None, None, False, msg
    return df, detect_schema(df), True, msg

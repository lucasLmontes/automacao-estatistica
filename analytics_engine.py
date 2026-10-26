from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from data_loader import detect_schema

MAX_TEXT_SNIPPET_LENGTH = 90
MAX_CONTEXT_SAMPLE_ROWS = 3
TOP_CATEGORIES_LIMIT = 7
TOP_CROSS_LIMIT = 5


def _sanitize_context_value(val: Any) -> Any:
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)[:19]
    if isinstance(val, (float, np.floating)):
        return round(float(val), 2)
    if isinstance(val, (int, np.integer)):
        return int(val)

    text = str(val).strip()
    return f"{text[:MAX_TEXT_SNIPPET_LENGTH]}..." if len(text) > MAX_TEXT_SNIPPET_LENGTH else text


def _extrair_amostra_contextual(df: pd.DataFrame, max_rows: int = MAX_CONTEXT_SAMPLE_ROWS) -> List[Dict[str, Any]]:
    return [
        {col: _sanitize_context_value(val) for col, val in row.items()}
        for _, row in df.head(max_rows).iterrows()
    ]


def _calcular_estatisticas_numericas(series: pd.Series) -> Dict[str, Any]:
    num_series = pd.to_numeric(series, errors="coerce").dropna()
    if num_series.empty:
        return {}

    q25 = float(num_series.quantile(0.25))
    q75 = float(num_series.quantile(0.75))
    iqr = q75 - q25
    outliers = int(((num_series < (q25 - 1.5 * iqr)) | (num_series > (q75 + 1.5 * iqr))).sum())

    return {
        "contagem_validos": len(num_series),
        "media": round(float(num_series.mean()), 2),
        "mediana": round(float(num_series.median()), 2),
        "desvio_padrao": round(float(num_series.std()), 2) if len(num_series) > 1 else 0.0,
        "minimo": round(float(num_series.min()), 2),
        "maximo": round(float(num_series.max()), 2),
        "soma": round(float(num_series.sum()), 2),
        "q25": round(q25, 2),
        "q75": round(q75, 2),
        "quantidade_outliers": outliers,
    }


def _derive_temporal_metric(df: pd.DataFrame, temporal_cols: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if len(temporal_cols) < 2:
        return df, None

    try:
        dt1 = pd.to_datetime(df[temporal_cols[0]], errors="coerce")
        dt2 = pd.to_datetime(df[temporal_cols[1]], errors="coerce")

        if dt2.mean() >= dt1.mean():
            start_dt, end_dt = dt1, dt2
            metric_name = f"duracao_{temporal_cols[1]}_vs_{temporal_cols[0]}_horas"
        else:
            start_dt, end_dt = dt2, dt1
            metric_name = f"duracao_{temporal_cols[0]}_vs_{temporal_cols[1]}_horas"

        diff_hours = (end_dt - start_dt).dt.total_seconds() / 3600.0
        if diff_hours.notnull().sum() > 0 and (diff_hours >= 0).sum() > 0:
            df_derived = df.copy()
            df_derived[metric_name] = diff_hours
            return df_derived, metric_name
    except Exception:
        pass

    return df, None


def _format_cross_metric_row(row: pd.Series, cat_col: str) -> Dict[str, Any]:
    return {
        "categoria": str(row[cat_col]),
        "media": round(float(row["mean"]), 2),
        "maximo": round(float(row["max"]), 2) if "max" in row else None,
        "ocorrencias": int(row["count"]),
    }


def _empty_payload() -> Dict[str, Any]:
    return {
        "status": "empty",
        "mensagem": "Nenhum dado disponível para análise estatística.",
        "total_registros": 0,
        "resumo_geral": {},
        "classificacao_colunas": {},
        "analise_categorica": {},
        "analise_numerica": {},
        "analise_cruzada": {},
        "amostra_contextual": [],
    }


def compute_dynamic_metrics(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if df is None or df.empty:
        return _empty_payload()

    if schema is None:
        schema = detect_schema(df)

    total_registros = len(df)
    total_colunas = len(df.columns)
    classificacoes = schema.get("classifications", {})

    col_cats = classificacoes.get("categorical", [])
    col_nums = list(classificacoes.get("numeric_metrics", []))
    col_texts = classificacoes.get("text", [])
    col_ids = classificacoes.get("identifiers", [])
    col_temps = classificacoes.get("temporal", [])

    total_cells = total_registros * total_colunas
    total_nulls = int(df.isnull().sum().sum())
    taxa_preenchimento = (
        round(((total_cells - total_nulls) / total_cells) * 100.0, 2)
        if total_cells > 0
        else 100.0
    )

    df_calc, derived_metric = _derive_temporal_metric(df, col_temps)
    if derived_metric and derived_metric not in col_nums:
        col_nums.append(derived_metric)

    analise_categorica: Dict[str, Any] = {}
    for col in col_cats:
        series = df_calc[col].dropna().astype(str).str.strip()
        if series.empty:
            continue

        counts = series.value_counts()
        total_valid = len(series)
        analise_categorica[col] = {
            "cardinalidade": len(counts),
            "total_validos": total_valid,
            "valor_predominante": str(counts.index[0]),
            "ocorrencias_predominante": int(counts.iloc[0]),
            "percentual_predominante": round((int(counts.iloc[0]) / total_valid) * 100.0, 2),
            "ranking_top": [
                {
                    "valor": str(k),
                    "ocorrencias": int(v),
                    "percentual": round((int(v) / total_valid) * 100.0, 2),
                }
                for k, v in counts.head(TOP_CATEGORIES_LIMIT).items()
            ],
        }

    analise_numerica: Dict[str, Any] = {
        col: _calcular_estatisticas_numericas(df_calc[col])
        for col in col_nums
        if col in df_calc.columns and not df_calc[col].dropna().empty
    }

    analise_cruzada: Dict[str, Any] = {}
    cat_foco = col_cats[0] if col_cats else None
    num_foco = col_nums[0] if col_nums else None

    if cat_foco and num_foco and cat_foco in df_calc.columns and num_foco in df_calc.columns:
        try:
            df_subset = df_calc[[cat_foco, num_foco]].dropna().copy()
            df_subset[num_foco] = pd.to_numeric(df_subset[num_foco], errors="coerce")
            df_subset = df_subset.dropna()

            if not df_subset.empty:
                grouped = (
                    df_subset.groupby(cat_foco)[num_foco]
                    .agg(["count", "mean", "max"])
                    .reset_index()
                    .sort_values(by="mean", ascending=False)
                )
                highest_mean = grouped.iloc[0]
                highest_volume = grouped.sort_values(by="count", ascending=False).iloc[0]

                analise_cruzada = {
                    "categoria_analisada": cat_foco,
                    "metrica_analisada": num_foco,
                    "segmento_maior_media": _format_cross_metric_row(highest_mean, cat_foco),
                    "segmento_maior_volume": _format_cross_metric_row(highest_volume, cat_foco),
                    "ranking_cruzado_top_5": [
                        _format_cross_metric_row(row, cat_foco)
                        for _, row in grouped.head(TOP_CROSS_LIMIT).iterrows()
                    ],
                }
        except Exception:
            pass

    if len(col_cats) >= 2 and not analise_cruzada:
        try:
            cat1, cat2 = col_cats[0], col_cats[1]
            cross = (
                df_calc.groupby([cat1, cat2])
                .size()
                .reset_index(name="ocorrencias")
                .sort_values(by="ocorrencias", ascending=False)
            )
            analise_cruzada = {
                "tipo": "intersecao_categorica",
                "dimensao_primaria": cat1,
                "dimensao_secundaria": cat2,
                "top_combinacoes": [
                    {
                        cat1: str(r[cat1]),
                        cat2: str(r[cat2]),
                        "ocorrencias": int(r["ocorrencias"]),
                        "percentual": round((int(r["ocorrencias"]) / total_registros) * 100.0, 2),
                    }
                    for _, r in cross.head(TOP_CROSS_LIMIT).iterrows()
                ],
            }
        except Exception:
            pass

    dimensao_destaque = None
    if col_cats and col_cats[0] in analise_categorica:
        cat_info = analise_categorica[col_cats[0]]
        dimensao_destaque = {
            "nome_coluna": col_cats[0],
            "valor_mais_frequente": cat_info["valor_predominante"],
            "volume_mais_frequente": cat_info["ocorrencias_predominante"],
            "percentual_volume": cat_info["percentual_predominante"],
            "cardinalidade_total": cat_info["cardinalidade"],
        }

    return {
        "status": "success",
        "total_registros": total_registros,
        "total_colunas": total_colunas,
        "taxa_preenchimento_global_pct": taxa_preenchimento,
        "classificacao_colunas": {
            "categoricas": col_cats,
            "metricas_numericas": col_nums,
            "textuais": col_texts,
            "identificadores": col_ids,
            "temporais": col_temps,
        },
        "dimensao_principal": dimensao_destaque,
        "analise_categorica": analise_categorica,
        "analise_numerica": analise_numerica,
        "analise_cruzada": analise_cruzada,
        "amostra_contextual": _extrair_amostra_contextual(df_calc),
    }


def format_dynamic_facts_for_prompt(facts: Dict[str, Any]) -> str:
    if not facts or facts.get("status") == "empty":
        return "Nenhum dado estatístico disponível na planilha."

    total = facts.get("total_registros", 0)
    total_cols = facts.get("total_colunas", 0)
    taxa_preench = facts.get("taxa_preenchimento_global_pct", 100.0)
    cols = facts.get("classificacao_colunas", {})
    cats = facts.get("analise_categorica", {})
    nums = facts.get("analise_numerica", {})
    cruzada = facts.get("analise_cruzada", {})

    linhas = [
        "ESTRUTURA DA PLANILHA:",
        f"- Total de registros: {total:,}",
        f"- Total de colunas: {total_cols} (Preenchimento global: {taxa_preench}%)",
        f"- Colunas Categóricas identificadas: {', '.join(cols.get('categoricas', [])) or 'Nenhuma'}",
        f"- Métricas Numéricas identificadas: {', '.join(cols.get('metricas_numericas', [])) or 'Nenhuma'}",
        f"- Colunas Textuais/Descritivas: {', '.join(cols.get('textuais', [])) or 'Nenhuma'}",
        f"- Identificadores/Chaves: {', '.join(cols.get('identificadores', [])) or 'Nenhum'}",
        f"- Colunas Temporais/Datas: {', '.join(cols.get('temporais', [])) or 'Nenhuma'}",
        "",
        "DISTRIBUIÇÕES CATEGÓRICAS PRINCIPAIS:",
    ]

    for col_name, info in list(cats.items())[:3]:
        linhas.append(
            f"• Atributo '{col_name}' ({info.get('cardinalidade', 0)} valores distintos): "
            f"Predominante: '{info.get('valor_predominante', 'N/A')}' com {info.get('ocorrencias_predominante', 0)} ocorrências ({info.get('percentual_predominante', 0)}%)."
        )
        ranking = info.get("ranking_top", [])[:3]
        top_str = ", ".join(f"'{r['valor']}': {r['ocorrencias']} ({r['percentual']}%)" for r in ranking)
        linhas.append(f"  Top valores: {top_str}")

    if nums:
        linhas.append("\nMÉTRICAS NUMÉRICAS CONSOLIDADAS:")
        for met_name, stats in list(nums.items())[:4]:
            linhas.append(
                f"• Métrica '{met_name}': Média = {stats.get('media', 0)}, Mediana = {stats.get('mediana', 0)}, "
                f"Mínimo = {stats.get('minimo', 0)}, Máximo = {stats.get('maximo', 0)}, Desvio Padrão = {stats.get('desvio_padrao', 0)}"
            )

    if cruzada:
        linhas.append("\nANÁLISE DE GARGALO / CRUZAMENTO DINÂMICO:")
        if "segmento_maior_media" in cruzada:
            seg = cruzada["segmento_maior_media"]
            linhas.append(
                f"• Maior média em '{cruzada.get('metrica_analisada')}': Segmento '{seg.get('categoria')}' "
                f"com média de {seg.get('media')} (Máx: {seg.get('maximo')}, Amostra: {seg.get('ocorrencias')} registros)."
            )
        elif "top_combinacoes" in cruzada:
            comb = cruzada["top_combinacoes"][0]
            dim1, dim2 = cruzada.get("dimensao_primaria"), cruzada.get("dimensao_secundaria")
            linhas.append(
                f"• Maior concentração combinada: {dim1}='{comb.get(dim1)}' + {dim2}='{comb.get(dim2)}' "
                f"representando {comb.get('ocorrencias')} registros ({comb.get('percentual')}% do volume total)."
            )

    return "\n".join(linhas)


format_facts_for_prompt = format_dynamic_facts_for_prompt

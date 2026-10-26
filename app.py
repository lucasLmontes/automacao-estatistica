"""Módulo 4: Interface Web Streamlit para Diagnóstico e Prescrição 5W2H."""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from data_loader import load_and_profile_data
from analytics_engine import compute_dynamic_metrics, format_dynamic_facts_for_prompt
from ai_service import AVAILABLE_MODELS, DEFAULT_MODEL, gerar_plano_5w2h, plano_para_dataframe, Plano5W2H

SESSION_DEFAULTS = {
    "df_validado": None,
    "schema_info": None,
    "fatos_estatisticos": None,
    "plano_5w2h": None,
    "ultimo_arquivo_id": None,
}


def _init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _render_schema_badges(classifications: Dict[str, list]) -> None:
    st.subheader("Smart Schema Detection")
    st.caption("Estrutura e tipos inferidos automaticamente da base carregada:")

    category_labels = {
        "categorical": "Categórica",
        "numeric_metrics": "Métrica",
        "text": "Textual",
        "identifiers": "Identificador",
        "temporal": "Temporal",
    }

    pills = [
        f"{label}: {col}"
        for cat, label in category_labels.items()
        for col in classifications.get(cat, [])
    ]
    if pills:
        st.pills(
            label="Colunas Detectadas",
            options=pills,
            default=pills,
            selection_mode="multi",
            disabled=True,
            label_visibility="collapsed",
        )


def _render_data_health(df: pd.DataFrame, schema_info: Dict[str, Any], facts: Dict[str, Any]) -> None:
    with st.expander("Diagnóstico de Qualidade dos Dados Brutos", expanded=False):
        total_cells = schema_info["total_rows"] * schema_info["total_columns"]
        total_nulls = int(df.isnull().sum().sum())
        fill_rate = facts.get("taxa_preenchimento_global_pct", 100.0)

        details = schema_info.get("details", {})
        cols_with_nulls = [c for c, d in details.items() if d.get("null_count", 0) > 0]
        cols_complete = schema_info["total_columns"] - len(cols_with_nulls)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taxa Global de Preenchimento", f"{fill_rate}%")
        with col2:
            st.metric("Células Nulas Detectadas", f"{total_nulls:,} de {total_cells:,}")
        with col3:
            st.metric("Colunas Íntegras (Sem Nulos)", f"{cols_complete} de {schema_info['total_columns']}")

        if cols_with_nulls:
            null_data = [
                {
                    "Coluna": col,
                    "Tipo Inferido": details[col].get("inferred_type", "N/A"),
                    "Valores Ausentes": details[col].get("null_count", 0),
                    "Percentual Nulos": f"{details[col].get('null_pct', 0.0)}%",
                }
                for col in cols_with_nulls
            ]
            st.write("**Detalhamento de Colunas com Valores Ausentes:**")
            st.dataframe(pd.DataFrame(null_data), use_container_width=True, hide_index=True)
        else:
            st.success("Todas as colunas estão 100% íntegras, sem registros ausentes.")

        st.info(
            "Tratamento Automático Aplicado: Linhas e colunas totalmente vazias foram removidas, "
            "espaços em branco foram higienizados e valores ausentes são desconsiderados de forma dinâmica "
            "nas métricas agregadas, prevenindo distorções estatísticas e garantindo integridade analítica."
        )


def _render_kpi_cards(facts: Dict[str, Any], total_columns: int) -> None:
    col1, col2, col3, col4 = st.columns(4)
    total_reg = facts.get("total_registros", 0)
    taxa_preench = facts.get("taxa_preenchimento_global_pct", 100.0)
    dim_destaque = facts.get("dimensao_principal")
    analise_nums = facts.get("analise_numerica", {})
    analise_cruzada = facts.get("analise_cruzada", {})

    with col1:
        st.metric(
            label="Total de Registros",
            value=f"{total_reg:,}",
            help=f"Preenchimento global: {taxa_preench}%",
        )

    with col2:
        if dim_destaque:
            st.metric(
                label=f"Predominância em '{dim_destaque['nome_coluna']}'",
                value=str(dim_destaque["valor_mais_frequente"])[:18],
                delta=f"{dim_destaque['percentual_volume']}% do volume",
                help=f"{dim_destaque['volume_mais_frequente']} registros de {total_reg}",
            )
        else:
            st.metric(label="Colunas Mapeadas", value=f"{total_columns}")

    with col3:
        if analise_nums:
            first_metric = list(analise_nums.keys())[0]
            stats = analise_nums[first_metric]
            friendly_name = first_metric.replace("duracao_", "Duração (").replace("_horas", "h)")
            st.metric(
                label=f"Média: {friendly_name[:22]}",
                value=f"{stats.get('media', 0):,}",
                help=f"Mediana: {stats.get('mediana', 0)} | Máx: {stats.get('maximo', 0)}",
            )
        else:
            cat_count = len(facts.get("classificacao_colunas", {}).get("categoricas", []))
            st.metric(label="Dimensões Categóricas", value=f"{cat_count} mapeadas")

    with col4:
        if analise_cruzada and "segmento_maior_media" in analise_cruzada:
            seg = analise_cruzada["segmento_maior_media"]
            st.metric(
                label=f"Maior Gargalo ({analise_cruzada.get('categoria_analisada', '')})",
                value=str(seg.get("categoria"))[:18],
                delta=f"Média: {seg.get('media')}h",
                delta_color="inverse",
                help=f"Maior valor médio encontrado na métrica '{analise_cruzada.get('metrica_analisada')}'.",
            )
        elif analise_cruzada and "top_combinacoes" in analise_cruzada:
            top_comb = analise_cruzada["top_combinacoes"][0]
            st.metric(
                label="Maior Concentração",
                value=f"{top_comb.get('ocorrencias', 0)} itens",
                delta=f"{top_comb.get('percentual', 0)}% base",
            )
        else:
            st.metric(label="Atributos Analisados", value=f"{total_columns} colunas")


def _render_visual_exploration(df: pd.DataFrame, facts: Dict[str, Any], cat_cols: list) -> None:
    with st.expander("Exploração Visual e Distribuições Dinâmicas", expanded=True):
        if not cat_cols:
            st.dataframe(df.head(20), use_container_width=True)
            return

        tab_dist, tab_num, tab_dados, tab_fatos = st.tabs(
            ["Distribuição Categórica", "Métricas Numéricas", "Amostra dos Dados", "Fatos da IA"]
        )

        with tab_dist:
            selected_col = st.selectbox("Selecione uma coluna para visualizar a distribuição:", options=cat_cols)
            if selected_col:
                counts = df[selected_col].dropna().astype(str).str.strip().value_counts().head(12)
                st.bar_chart(counts)

        with tab_num:
            analise_nums = facts.get("analise_numerica", {})
            if analise_nums:
                df_metricas = pd.DataFrame.from_dict(analise_nums, orient="index")
                st.dataframe(df_metricas, use_container_width=True)
            else:
                st.info("Nenhuma métrica numérica contínua detectada no arquivo.")

        with tab_dados:
            st.dataframe(df.head(20), use_container_width=True)

        with tab_fatos:
            st.code(format_dynamic_facts_for_prompt(facts), language="text")


def _render_action_cards(plano: Plano5W2H) -> None:
    with st.expander("Detalhamento Individual por Ação Prescritiva"):
        for i, acao in enumerate(plano.acoes_5w2h, start=1):
            st.subheader(f"Ação #{i}: {acao.o_que}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Por que fazer (Why):** {acao.por_que}")
                st.write(f"**Onde aplicar (Where):** {acao.onde}")
                st.write(f"**Prazo (When):** {acao.quando}")
            with col_b:
                st.write(f"**Responsável (Who):** {acao.quem}")
                st.write(f"**Como executar (How):** {acao.como}")
                st.write(f"**Custo / Esforço (How Much):** {acao.quanto}")
            if i < len(plano.acoes_5w2h):
                st.divider()


def _render_prescriptive_plan(plano: Plano5W2H) -> None:
    st.subheader("Diagnóstico Executivo da Operação")
    st.info(plano.resumo_executivo, icon="📋")

    df_5w2h = plano_para_dataframe(plano)

    st.subheader("Matriz de Ações Prescritivas (5W2H)")
    st.dataframe(df_5w2h, use_container_width=True, hide_index=True)

    _render_action_cards(plano)

    st.subheader("Exportar Resultados")
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="Baixar Plano em JSON",
            data=plano.model_dump_json(indent=2),
            file_name="plano_prescritivo_5w2h.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_down2:
        st.download_button(
            label="Baixar Tabela em CSV",
            data=df_5w2h.to_csv(index=False, encoding="utf-8-sig"),
            file_name="plano_prescritivo_5w2h.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Pipeline Inteligente | Diagnóstico Prescritivo 5W2H",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    with st.sidebar:
        st.title("Controles do Pipeline")
        st.write("Carregue qualquer base de dados tabular para acionar o diagnóstico dinâmico.")

        uploaded_file = st.file_uploader(
            label="Carregar Base de Dados",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Suporta qualquer arquivo CSV ou Excel. O esquema e as colunas serão inferidos dinamicamente.",
        )

        st.divider()
        st.subheader("Configuração da IA")

        custom_api_key = st.text_input(
            label="Chave Gemini (Opcional)",
            type="password",
            placeholder="Padrão: lida do arquivo .env",
            help="Caso não informada, a chave configurada no .env será utilizada automaticamente.",
        )

        selected_model = st.selectbox(
            label="Modelo Gemini",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
            help="Modelo preferencial para geração estruturada do plano 5W2H.",
        )

        st.divider()
        st.caption("Pipeline Agnóstico Universal • Smart Schema Detection v2.0")

    st.title("Pipeline Operacional: Diagnóstico e Prescrição 5W2H")
    st.caption("Leitura Universal & Smart Schema Detection → Motor Estatístico Determinístico → Prescrição 5W2H com IA")

    if uploaded_file is None:
        st.info(
            "**Bem-vindo ao Pipeline Universal de Diagnóstico e Prescrição 5W2H**\n\n"
            "Para começar, utilize a barra lateral esquerda para enviar sua planilha em formato `.csv`, `.tsv` ou `.xlsx`.\n\n"
            "O sistema inspecionará automaticamente o arquivo, classificará os tipos de atributos "
            "(Smart Schema Detection), calculará as estatísticas essenciais e alimentará o modelo de IA para prescrever "
            "o plano de ação 5W2H baseado nos dados reais."
        )
        return

    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.ultimo_arquivo_id != current_file_id:
        st.session_state.ultimo_arquivo_id = current_file_id
        st.session_state.plano_5w2h = None

    df_loaded, schema_detected, is_valid, validation_msg = load_and_profile_data(uploaded_file)
    if not is_valid or df_loaded is None or schema_detected is None:
        st.error(validation_msg)
        st.stop()

    st.session_state.df_validado = df_loaded
    st.session_state.schema_info = schema_detected
    st.success(validation_msg)

    facts = compute_dynamic_metrics(df_loaded, schema=schema_detected)
    st.session_state.fatos_estatisticos = facts

    _render_schema_badges(schema_detected["classifications"])
    _render_data_health(df_loaded, schema_detected, facts)

    st.subheader("Diagnóstico Determinístico da Base")
    _render_kpi_cards(facts, total_columns=len(df_loaded.columns))

    cat_cols = schema_detected["classifications"].get("categorical", [])
    _render_visual_exploration(df_loaded, facts, cat_cols)

    st.divider()
    st.subheader("Prescrição 5W2H Adaptativa com Inteligência Artificial")
    st.write(
        "A IA inspecionará o payload estatístico real acima para prescrever um plano de ação "
        "ancorado na estrutura e evidências específicas desta planilha."
    )

    if st.button("Gerar Plano de Ação 5W2H", type="primary", use_container_width=True):
        with st.spinner("Analisando evidências estatísticas e sintetizando ações 5W2H via Google Gemini..."):
            plano, success, ai_msg = gerar_plano_5w2h(
                fatos_estatisticos=facts,
                api_key=custom_api_key.strip() or None,
                modelo_preferencial=selected_model,
            )
            if not success or plano is None:
                st.error(ai_msg)
            else:
                st.session_state.plano_5w2h = plano
                st.success("Plano Prescritivo 5W2H gerado e validado com sucesso.")

    if st.session_state.plano_5w2h is not None:
        _render_prescriptive_plan(st.session_state.plano_5w2h)


if __name__ == "__main__":
    main()

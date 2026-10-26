import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field

AVAILABLE_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
)
DEFAULT_MODEL = "gemini-2.5-flash"

AUTH_ERROR_KEYWORDS = ("api_key", "permission", "unauthenticated")
QUOTA_ERROR_KEYWORDS = ("quota", "429", "resourceexhausted")
UNAVAILABLE_ERROR_KEYWORDS = ("503", "unavailable")


class Item5W2H(BaseModel):
    o_que: str = Field(description="Ação prescritiva recomendada (O que fazer).")
    por_que: str = Field(description="Justificativa técnica ou causa-raiz vinculada aos dados estatísticos (Por que fazer).")
    onde: str = Field(description="Setor, fila, sistema ou componente operacional afetado (Onde fazer).")
    quando: str = Field(description="Prazo ou cronograma estimado de implantação (Quando fazer).")
    quem: str = Field(description="Responsável ou papel organizacional executor (Quem fará).")
    como: str = Field(description="Procedimento ou método prático de execução (Como fazer).")
    quanto: str = Field(description="Custo estimado, esforço ou alocação de recursos (Quanto custará).")


class Plano5W2H(BaseModel):
    resumo_executivo: str = Field(description="Diagnóstico conciso da operação em 1 ou 2 frases, citando os dados observados.")
    acoes_5w2h: List[Item5W2H] = Field(description="Lista de 3 a 5 ações prescritivas essenciais no modelo 5W2H.")


def _carregar_env_local() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


def obter_cliente_gemini(api_key: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    _carregar_env_local()
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None, (
            "Chave de API do Google Gemini não encontrada. "
            "Defina a variável GEMINI_API_KEY no arquivo '.env' ou informe-a na aplicação."
        )

    try:
        from google import genai
        return genai.Client(api_key=key), None
    except ImportError:
        return None, "Biblioteca 'google-genai' não está instalada. Execute: pip install google-genai"
    except Exception as exc:
        return None, f"Erro ao inicializar cliente Google Gemini: {exc}"


def construir_prompts(fatos_estatisticos: Dict[str, Any]) -> Tuple[str, str]:
    system_prompt = (
        "Você é um Engenheiro Sênior de Otimização de Processos e Inteligência Operacional focado em melhoria contínua "
        "e resolução sistemática de problemas. "
        "Sua missão é inspecionar o esquema, as métricas e as distribuições reais da base de dados e prescrever um "
        "Plano de Ação 5W2H altamente técnico, pragmático, mensurável e direcionado aos principais gargalos, "
        "categorias mais volumosas ou anomalias evidenciadas nos dados. "
        "Responda estritamente no esquema estruturado fornecido (Plano5W2H)."
    )

    fatos_json = json.dumps(fatos_estatisticos, indent=2, ensure_ascii=False)
    user_prompt = f"""EVIDÊNCIAS ESTATÍSTICAS E ESTRUTURA REAL DA BASE DE DADOS:
```json
{fatos_json}
```

DIRETRIZES OBRIGATÓRIAS PARA O DIAGNÓSTICO E PLANO 5W2H:
1. Resumo Executivo: Sintetize o diagnóstico da operação em até 2 frases diretas, citando nominalmente os atributos, números, categorias dominantes e métricas reais observadas na planilha.
2. Ações Prescritivas 5W2H: Formule de 3 a 5 ações pragmáticas e prioritárias focadas em:
   - Atacar a categoria, segmento ou serviço com maior volume de ocorrências ou maior gargalo/tempo médio.
   - Tratar a causa-raiz dos desvios, falhas recorrentes ou reincidências indicadas nas métricas ou exemplos.
   - Estabelecer padronização operacional, prevenção de incidentes e governança dos processos.
3. Especificidade Técnica: Preencha cada campo metodológico com rigor:
   - 'o_que' (What): Ação prescritiva clara e objetiva.
   - 'por_que' (Why): Justificativa causal amarrada aos dados quantitativos observados.
   - 'onde' (Where): Sistema, setor, fila ou componente operacional identificado nos dados.
   - 'quando' (When): Prazo estimado realista (ex: 'Imediato (15 dias)', 'Sprint 2 (30 dias)').
   - 'quem' (Who): Papel organizacional responsável (ex: 'Líder Técnico de Suporte', 'Especialista em Sistemas').
   - 'como' (How): Procedimento técnico ou plano prático de execução.
   - 'quanto' (How Much): Estimativa de custo, esforço ou alocação (ex: 'Baixo custo operacional / 20h equipe', 'R$ 0 / Ajuste de configuração').
"""
    return system_prompt, user_prompt


def _classify_gemini_error(exc: Exception) -> Optional[str]:
    err_str = str(exc).lower()
    if any(k in err_str for k in AUTH_ERROR_KEYWORDS):
        return "Falha de autenticação: A chave da API do Gemini informada é inválida ou expirou."
    if any(k in err_str for k in QUOTA_ERROR_KEYWORDS):
        return "Limite de requisições ou cota da API do Gemini atingido. Aguarde alguns instantes."
    return None


def gerar_plano_5w2h(
    fatos_estatisticos: Dict[str, Any],
    api_key: Optional[str] = None,
    modelo_preferencial: str = DEFAULT_MODEL,
) -> Tuple[Optional[Plano5W2H], bool, str]:
    if not fatos_estatisticos or fatos_estatisticos.get("status") == "empty":
        return None, False, "Não há fatos estatísticos válidos para processamento da IA."

    client, error_msg = obter_cliente_gemini(api_key)
    if error_msg or client is None:
        return None, False, error_msg or "Falha de autenticação na API do Gemini."

    from google.genai import types

    system_prompt, user_prompt = construir_prompts(fatos_estatisticos)
    candidate_models = [modelo_preferencial] + [m for m in AVAILABLE_MODELS if m != modelo_preferencial]
    last_error: Optional[Exception] = None

    for model in candidate_models:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=Plano5W2H,
                        temperature=0.2,
                    ),
                )

                if hasattr(response, "parsed") and isinstance(response.parsed, Plano5W2H):
                    return response.parsed, True, "Plano 5W2H gerado com sucesso pela IA."

                if hasattr(response, "text") and response.text:
                    return Plano5W2H.model_validate_json(response.text), True, "Plano 5W2H gerado e validado com sucesso."

            except Exception as exc:
                last_error = exc
                fatal_message = _classify_gemini_error(exc)
                if fatal_message:
                    return None, False, fatal_message

                if any(k in str(exc).lower() for k in UNAVAILABLE_ERROR_KEYWORDS):
                    time.sleep(1.5 * attempt)
                    continue

                break

    msg = str(last_error)
    if any(k in msg.lower() for k in UNAVAILABLE_ERROR_KEYWORDS):
        return None, False, "Os servidores do Google Gemini estão com alta demanda temporária (503). Tente novamente em instantes."

    return None, False, f"Falha na comunicação com a IA: {msg}"


def plano_para_dataframe(plano: Plano5W2H) -> pd.DataFrame:
    if not plano or not plano.acoes_5w2h:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "Item": f"#{idx}",
            "O que fazer (What)": acao.o_que,
            "Por que fazer (Why)": acao.por_que,
            "Onde (Where)": acao.onde,
            "Quando (When)": acao.quando,
            "Quem (Who)": acao.quem,
            "Como fazer (How)": acao.como,
            "Quanto custa (How Much)": acao.quanto,
        }
        for idx, acao in enumerate(plano.acoes_5w2h, start=1)
    ])

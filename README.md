# Pipeline Analítico e Prescritivo Agnóstico

## Visão Geral do Sistema

O sistema consiste em um pipeline analítico e prescritivo híbrido, agnóstico a esquemas de dados tabulares. A arquitetura separa o processamento de agregação e cálculo estatístico — executado de forma determinística em Python — da síntese preditiva e formulação de planos de ação no padrão 5W2H, delegada ao modelo de linguagem Google Gemini via Structured Outputs.

Esta abordagem híbrida elimina o risco de alucinações numéricas e erros estatísticos inerentes às arquiteturas puramente generativas, garantindo exatidão matemática sobre bases de dados de qualquer domínio corporativo.

## Arquitetura e Módulos do Sistema

O projeto é estruturado em quatro módulos independentes com responsabilidades desacopladas:

1. **Ingestão e Perfilamento (`data_loader.py`)**:

   * Leitura de arquivos `.csv`, `.tsv`, `.xlsx` e `.xls` com tentativa automática de diferentes encodings (UTF-8, Latin-1, CP1252) e delimitadores.

   * Higienização de cabeçalhos e remoção de registros nulos.

   * Smart Schema Detection: Classificação autônoma dos atributos em categóricos, métricas numéricas, identificadores, temporais e textuais.

2. **Motor Estatístico Determinístico (`analytics_engine.py`)**:

   * Cálculo de estatísticas descritivas (média, mediana, desvio-padrão, quantis e detecção de outliers via método IQR).

   * Apuração de cardinalidade e distribuições de frequência em variáveis categóricas.

   * Derivação automática de métricas de tempo decorrido e cruzamento bidimensional para identificação de gargalos operacionais.

3. **Ponte Cognitiva e Prescrição (`ai_service.py`)**:

   * Formatação e transmissão do payload de fatos estatísticos ao Google Gemini.

   * Validação estrita do contrato de resposta (matriz 5W2H) utilizando Pydantic e o recurso de Structured Outputs da API.

4. **Interface do Usuário e Apresentação (`app.py`)**:

   * Aplicação web em Streamlit com painel de qualidade de dados (Smart Data Health).

   * Renderização do diagnóstico executivo, cartões de KPIs e tabela 5W2H.

   * Exportação multiformato (JSON, CSV e relatório HTML formatado para impressão em PDF).

## Requisitos de Sistema e Dependências

### Pré-requisitos

* **Python**: Versão 3.10 ou superior.

* **Gerenciador de Pacotes**: `pip`.

* **Chave de API do Google Gemini**: Necessária para a execução do serviço de prescrição por IA.

### Dependências Fundamentais

* **Pandas (v2.0+)**: Manipulação de dados tabulares, leitura de arquivos e agregações determinísticas.

* **NumPy**: Operações vetoriais e cálculos matemáticos de alta performance.

* **OpenPyXL / xlrd**: Leitura de planilhas do Microsoft Excel (`.xlsx` / `.xls`).

* **Pydantic (v2.0+)**: Definição dos contratos de dados (schemas) e validação de respostas estruturadas da IA.

* **google-genai**: SDK oficial para comunicação autenticada com a API do Google Gemini.

* **Streamlit**: Interface web reativa e gerenciamento do estado da aplicação.

* **python-dotenv / python-decouple**: Leitura segura de variáveis de ambiente.

## Guia de Instalação e Configuração

### 1. Clonar o Repositório

```
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

```

### 2. Criar e Ativar um Ambiente Virtual

No Linux/macOS:

```
python3 -m venv venv
source venv/bin/activate

```

No Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1

```

### 3. Instalar as Dependências

Execute o comando abaixo para instalar todas as bibliotecas necessárias:

```
pip install pandas numpy openpyxl xlrd pydantic google-genai streamlit python-dotenv

```

Caso exista um arquivo `requirements.txt` no repositório, você pode instalar diretamente com:

```
pip install -r requirements.txt

```

### 4. Configurar as Variáveis de Ambiente

Crie um arquivo chamado `.env` na raiz do projeto com o seguinte conteúdo:

```
GEMINI_API_KEY=sua_chave_api_aqui

```

Nota: Opcionalmente, a chave de API também poderá ser informada diretamente na barra lateral da aplicação Streamlit durante a execução.

## Como Executar o Projeto

Com o ambiente virtual ativado e as dependências instaladas, inicie a aplicação Streamlit através do comando:

```
streamlit run app.py

```

A aplicação será aberta automaticamente no seu navegador padrão no endereço local:
`http://localhost:8501`

## Estrutura do Repositório

```
.
├── app.py                # Interface visual e fluxo principal (Streamlit)
├── data_loader.py        # Módulo de ingestão e perfilamento de dados
├── analytics_engine.py   # Motor estatístico determinístico
├── ai_service.py         # Integração com Google Gemini e schemas Pydantic
├── .env.example          # Modelo de configuração das variáveis de ambiente
├── .gitignore            # Regras para exclusão de arquivos sensíveis e temporários
└── README.md             # Documentação técnica do projeto

```

## Segurança e Privacidade dos Dados

* **Gestão de Chaves de API**: As credenciais não são expostas no código-fonte. A leitura ocorre por variáveis de ambiente ou inserção mascarada na interface.

* **Privacidade dos Dados Brutos**: As planilhas carregadas permanecem restritas à memória local do servidor de aplicação. Apenas resumos e agregados estatísticos anônimos são enviados à API da IA.

* **Versionamento**: Arquivos sensíveis (`.env`), ambientes virtuais (`venv/`) e caches de compilação (`__pycache__/`) são ignorados pelo Git via `.gitignore`.

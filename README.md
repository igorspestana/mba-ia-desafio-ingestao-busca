# Desafio MBA Engenharia de Software com IA - Full Cycle

## Sistema de Ingestão e Busca Semântica com IA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17+-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)

Sistema de RAG (Retrieval-Augmented Generation) que permite ingerir documentos PDF e realizar buscas semânticas inteligentes usando LangChain, PostgreSQL com pgVector e OpenAI.

## 📋 Sobre o Projeto

Este projeto implementa um sistema completo de ingestão e busca semântica que:

- **Ingere documentos PDF** dividindo-os em chunks otimizados
- **Armazena embeddings vetoriais** no PostgreSQL com extensão pgVector
- **Realiza buscas semânticas** inteligentes baseadas no conteúdo dos documentos
- **Fornece respostas contextuais** usando LLM (Large Language Model) da OpenAI
- **Garante precisão** respondendo apenas com base no conteúdo ingerido

### 🎯 Funcionalidades

- ✅ Ingestão automática de documentos PDF
- ✅ Divisão inteligente de texto em chunks com overlap
- ✅ Geração de embeddings vetoriais (OpenAI text-embedding-3-small)
- ✅ Armazenamento vetorial no PostgreSQL + pgVector
- ✅ Interface CLI interativa para consultas
- ✅ Busca semântica com 10 resultados mais relevantes
- ✅ Respostas contextuais baseadas apenas no conteúdo ingerido
- ✅ Validação de perguntas fora do contexto

## 🏗️ Arquitetura

### Fluxo de Ingestão (Processamento do PDF)
```
📄 PDF Document
    ↓
🔍 PyPDFLoader (extrai texto)
    ↓
✂️ RecursiveCharacterTextSplitter
    (chunk_size=250, overlap=70)
    ↓
🧠 OpenAI Embeddings
    (text-embedding-3-small)
    ↓
💾 PostgreSQL + pgVector
    (Collection Storage)
```

### Fluxo de Consulta (Busca e Resposta)
```
❓ User Question
    ↓
🧠 Question Embedding
    (text-embedding-3-small)
    ↓
🔍 Vector Similarity Search
    (k=10 resultados mais relevantes)
    ↓
📝 Context Assembly
    (concatena top resultados)
    ↓
🤖 OpenAI GPT-4o-mini
    (Temperature=0)
    ↓
💬 Contextual Answer
    (baseada apenas no conteúdo do PDF)
```

### Diagrama de Componentes
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   📄 PDF        │    │   🧠 OpenAI      │    │   💾 Database   │
│   Document      │───▶│   Embeddings     │───▶│   PostgreSQL    │
│                 │    │   (text-embedding│    │   + pgVector    │
└─────────────────┘    │    -3-small)     │    │                 │
                       └──────────────────┘    └─────────────────┘
                                │                        ▲
                                │                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   💬 User       │    │   🔍 Vector      │    │   📝 Context    │
│   Question      │───▶│   Search         │◀───│   Assembly      │
│                 │    │   (k=10)         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   🤖 OpenAI      │
                       │   GPT-4o-mini    │
                       │   (Temperature=0)│
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   💬 Contextual  │
                       │   Answer         │
                       └──────────────────┘
```

### 🛠️ Tecnologias Utilizadas

- **Python 3.10+** - Linguagem principal
- **LangChain** - Framework para aplicações com LLM
- **PostgreSQL 17** - Banco de dados relacional
- **pgVector** - Extensão para busca vetorial
- **OpenAI API** - Embeddings e LLM
- **Docker & Docker Compose** - Containerização
- **PyPDF** - Processamento de PDFs

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- Chave de API da OpenAI

### 1. Clone o repositório

```bash
git clone https://github.com/igorspestana/mba-ia-desafio-ingestao-busca
cd mba-ia-desafio-ingestao-busca
```

### 2. Instale o módulo de ambiente virtual (caso ainda não tenha)

```bash
sudo apt install python3-venv
```

### 3. Configure o ambiente virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Linux/Mac)
source venv/bin/activate

# Ativar ambiente virtual (Windows)
venv\Scripts\activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto e defina as variáveis:

```env
GOOGLE_API_KEY=
GOOGLE_EMBEDDING_MODEL='models/embedding-001'
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
DATABASE_URL=
PG_VECTOR_COLLECTION_NAME=
PDF_PATH=reports/spreadsheet_report.pdf
```

### 6. Inicie o banco de dados

```bash
docker compose up -d
```

Aguarde alguns segundos para o PostgreSQL inicializar completamente.

## 📖 Como Usar

### 1. Execute a ingestão do documento

```bash
python src/ingest.py
```

Este comando irá:
- Carregar o PDF especificado em `PDF_PATH`
- Dividir o documento em chunks de 250 caracteres com overlap de 70
- Gerar embeddings para cada chunk usando `text-embedding-3-small`
- Armazenar no banco de dados PostgreSQL com pgVector

### 2. Inicie o chat interativo

```bash
python src/chat.py
```

### 3. Faça suas perguntas

```
🤖 CHAT DE BUSCA SEMÂNTICA
============================================================
Digite 'sair' para encerrar o chat.
============================================================
✅ Chat inicializado com sucesso!

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
🔍 Buscando informações...
RESPOSTA: O faturamento foi de 10 milhões de reais.
------------------------------------------------------------

PERGUNTA: Quantos clientes temos em 2024?
🔍 Buscando informações...
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
------------------------------------------------------------
```

## 🔧 Comandos Úteis

### Gerenciamento do Ambiente Virtual

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Desativar ambiente virtual
deactivate

# Verificar dependências instaladas
pip list
```

### Monitoramento do Banco de Dados

```bash
# Verificar se o banco está rodando
docker ps

# Verificar tabelas existentes
docker exec postgres_rag psql -U postgres -d rag -c "\dt"

# Verificar quantidade de embeddings
docker exec postgres_rag psql -U postgres -d rag -c "SELECT COUNT(*) FROM langchain_pg_embedding;"

# Verificar coleções
docker exec postgres_rag psql -U postgres -d rag -c "SELECT * FROM langchain_pg_collection;"

# Verificar estrutura das tabelas
docker exec postgres_rag psql -U postgres -d rag -c "\d langchain_pg_collection"
docker exec postgres_rag psql -U postgres -d rag -c "\d langchain_pg_embedding"
```

### Análise de Embeddings por Coleção

```bash
# Ver quantos embeddings cada coleção possui
docker exec postgres_rag psql -U postgres -d rag -c "SELECT c.name, COUNT(e.id) as embedding_count FROM langchain_pg_collection c LEFT JOIN langchain_pg_embedding e ON c.uuid = e.collection_id GROUP BY c.name, c.uuid;"
```

## 🐛 Troubleshooting

### Problemas Comuns

**Erro: "Environment variable OPENAI_API_KEY is not set"**
- Verifique se o arquivo `.env` existe e contém a chave da OpenAI
- Certifique-se de que o arquivo está na raiz do projeto

**Erro: "PDF file not found"**
- Verifique se o caminho em `PDF_PATH` no `.env` está correto
- Certifique-se de que o arquivo PDF existe no local especificado

**Erro: "Connection refused" no banco de dados**
- Verifique se o Docker está rodando: `docker ps`
- Reinicie os containers: `docker compose down && docker compose up -d`

**Erro: "No chunks found"**
- Verifique se o PDF não está corrompido
- Tente com um PDF diferente para teste

### Logs e Debug

```bash
# Ver logs do PostgreSQL
docker logs postgres_rag
```

## 📁 Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── docker-compose.yml          # Configuração do PostgreSQL + pgVector
├── requirements.txt            # Dependências Python
├── .env                       # Variáveis de ambiente (criar)
├── README.md                  # Este arquivo
├── reports/
│   └── spreadsheet_report.pdf # Documento para ingestão
├── src/
│   ├── ingest.py             # Script de ingestão do PDF
│   ├── search.py             # Lógica de busca semântica
│   ├── chat.py               # Interface CLI interativa
│   └── __pycache__/          # Cache Python (gerado automaticamente)
├── doc/
│   └── overview.md           # Documentação técnica detalhada
└── venv/                     # Ambiente virtual Python (gerado automaticamente)
    ├── bin/                  # Scripts de ativação
    ├── lib/                  # Bibliotecas instaladas
    └── pyvenv.cfg            # Configuração do ambiente virtual
```

## 📄 Licença

Este projeto é parte do desafio do MBA em Engenharia de Software com IA - Full Cycle.

## 🔗 Links Úteis

- [LangChain Documentation](https://python.langchain.com/)
- [pgVector Documentation](https://github.com/pgvector/pgvector)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
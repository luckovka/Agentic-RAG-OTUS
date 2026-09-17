# Agentic-RAG-OTUS
Project for LLM_engineer_05_2026_course

**Задача** 
построить интерактивный учебный ассистент, который отвечает по базе материалов ML/LLM-инженера, приводит источники, умеет пользоваться инструментами для вычислений/проверки, а при недостаточном качестве ответа повторяет поиск. 

**Стратегия развития проекта** 
v1 — текстовый RAG
v2 — Agentic RAG + tools + reviewer
v3 — Telegram bot
v4 — voice input/output
v5 — image/VLM support

**MVP**
- загрузка PDF / TXT / MD;
- извлечение текста и базовых metadata;
- chunking;
- embeddings;
- Qdrant;
- semantic top-K retrieval;
- генерация ответа строго по найденному контексту;
- ответ "не знаю", если информации в базе нет;
- указание источника ответа

Documents
   ↓
Parser
   ↓
Chunking + metadata
   ↓
Embeddings
   ↓
Qdrant
   ↓
Retriever
   ↓
Generator
   ↓
Answer + source

## Must-have к защите 
- advanced RAG;
- citations;
- Langfuse;
- Docker / docker-compose;
- README;
- demo 3–5 минут;
- запуск одной командой

## Should-have 
- router, 
- Python tool, 
- reviewer, 
- prompt-injection guard, 
- evaluation

## Nice-to-have 

- Telegram

## The future
- voice input/output, 
- VLM, 
- более серьёзные evals и нагрузочное тестирование

**Целевая архитектура**

                 DATA / INGESTION
                       │
        PDF / TXT / MD / HTML
                       │
                Parser / cleanup
                       │
             structural chunking
                       │
                    metadata
          source / page / header
                       │
                       ↓
                 Vector DB
                   Qdrant
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Vector search                 BM25 search
        │                             │
        └────────── Hybrid ───────────┘
                       │
                    Reranker
                       │
                       ↓
User ─────────────→ Query Router
                       │
           ┌───────────┼────────────┐
           ↓           ↓            ↓
        RAG         Python Tool    Direct
           │
           ↓
     Injection Guard
           │
           ↓
        Generator
           │
           ↓
        Reviewer
           │
       ┌───┴────┐
       │        │
      OK       FAIL
       │        │
       ↓        └──────────→ Retrieval again
 Answer + citations
       │
       ↓
    FastAPI
       │
       ↓
      UI


           OBSERVABILITY
                │
             Langfuse
      traces / latency / tokens

**Обвязка**
Docker / docker-compose
.env
requirements.txt or uv.lock
README.md
tests / evaluation
GitHub

**Критерии готовности проекта**
- запускается одной командой;
- код разбит на модули;
- секреты локально в .env; .env добавлен в .gitignore; в репозитории лежит .env.example без реальных ключей.
- есть Docker;
- есть Langfuse traces и latency;
- RAG отвечает по базе и умеет говорить «не знаю»;
- citations работают;
- в репозитории лежит тестовый корпус/примеры;
- README объясняет архитектуру, запуск и usage;
- записано видео 3–5 минут.

**Бонус-фишки**
RAGAS / LLM-as-a-judge evaluation
GitHub Actions

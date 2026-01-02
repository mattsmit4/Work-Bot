# ST-Bot Development Guide

## Project Overview

ST-Bot is a StarTech.com Product Assistant chatbot powered by a 4,178 product Excel catalog. It helps customers find the right cables, adapters, docks, and connectivity products through natural language queries.

**Tech Stack:** Python, Streamlit, LangChain, OpenAI, Pinecone

---

## Architecture

### Clean 3-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│  UI Layer (app_startech.py)                             │
│  - Streamlit interface                                  │
│  - Session management                                   │
│  - ~280 lines                                           │
├─────────────────────────────────────────────────────────┤
│  Orchestration Layer (core/orchestrator.py)             │
│  - Query processing coordination                        │
│  - Intent → Handler routing                             │
│  - State persistence                                    │
├─────────────────────────────────────────────────────────┤
│  Handler Layer (handlers/)                              │
│  - One handler per intent type                          │
│  - Clean, testable, focused                             │
├─────────────────────────────────────────────────────────┤
│  Core Layer (core/, llm/, ui/, config/)                 │
│  - Business logic, search, LLM features                 │
│  - Pure Python, no framework dependencies               │
└─────────────────────────────────────────────────────────┘
```

### Query Flow

```
User Query
    │
    ▼
app_startech.py ──► orchestrator.process_query()
                        │
                        ├── 1. Load state (guidance, questions)
                        ├── 2. Classify intent
                        ├── 3. Apply domain rules
                        ├── 4. Route to handler
                        ├── 5. Execute handler
                        ├── 6. Save state & log
                        │
                        ▼
                   handlers/*.py ──► HandlerResult
                        │
                        ▼
                   Response to user
```

---

## Project Structure

```
ST-Bot/
├── app_startech.py              # Streamlit UI (~280 lines)
│
├── core/                        # Business logic
│   ├── orchestrator.py         # Query processing coordinator
│   ├── context.py              # Data models (Intent, Product, etc.)
│   ├── intent.py               # Intent classification (13 types)
│   ├── filters.py              # Extract filters from queries
│   ├── search.py               # Cascading search strategy
│   ├── guidance.py             # Setup guidance flows
│   ├── product_validator.py    # Product validation
│   ├── api_retry.py            # API retry logic
│   └── structured_logging.py   # Logging utilities
│
├── handlers/                    # Intent handlers
│   ├── base.py                 # BaseHandler, HandlerContext, HandlerResult
│   ├── greeting.py             # greeting, farewell
│   ├── blocked.py              # warranty, pricing, install_help, etc.
│   ├── sku.py                  # explicit_sku
│   ├── search.py               # new_search, feature_search_accept
│   ├── guidance.py             # setup_guidance, setup_followup
│   └── followup.py             # multi_followup, single_followup, etc.
│
├── llm/                         # LLM-powered features
│   ├── domain_rules.py         # Business rules
│   ├── device_inference.py     # Infer connectors from devices
│   ├── product_ranker.py       # Rank search results
│   ├── response_builder.py     # Build responses
│   ├── query_analyzer.py       # Analyze query complexity
│   ├── followup_handler.py     # Follow-up questions
│   ├── technical_question_handler.py
│   └── prompts.py              # System prompts
│
├── ui/                          # UI utilities
│   ├── state.py                # Session + guidance persistence
│   ├── responses.py            # Response formatting
│   └── logging.py              # Conversation logging
│
├── config/                      # Configuration
│   ├── patterns.py             # Regex patterns
│   └── synonyms.py             # Term synonyms
│
├── tests/                       # 423 unit tests
├── data/                        # Static data files
└── logs/                        # Runtime logs
```

---

## Handler Architecture

Each intent type has a dedicated handler class:

| Handler | File | Intents |
|---------|------|---------|
| `GreetingHandler` | greeting.py | greeting |
| `FarewellHandler` | greeting.py | farewell |
| `InstallHelpHandler` | blocked.py | install_help |
| `WarrantyHandler` | blocked.py | warranty_question |
| `PricingHandler` | blocked.py | pricing_question |
| `ImpossibleProductHandler` | blocked.py | impossible_product |
| `OutOfScopeHandler` | blocked.py | out_of_scope |
| `ExplicitSKUHandler` | sku.py | explicit_sku |
| `NewSearchHandler` | search.py | new_search |
| `FeatureSearchHandler` | search.py | feature_search_accept |
| `SetupGuidanceHandler` | guidance.py | setup_guidance |
| `SetupFollowupHandler` | guidance.py | setup_followup |
| `FollowupHandler` | followup.py | multi_followup, single_followup, etc. |

### Adding a New Handler

1. Create handler class extending `BaseHandler`
2. Implement `handle(ctx: HandlerContext) -> HandlerResult`
3. Register in `core/orchestrator.py` HANDLERS dict
4. Add tests

---

## Development Principles

### Code Quality

- **SIMPLICITY FIRST**: Every change should impact as little code as possible
- **NO LAZY FIXES**: Find the root cause and fix it properly
- **COMPLETE IMPLEMENTATIONS**: Finish what you start
- **PRODUCTION QUALITY**: Code should be ready to deploy
- **DIAGNOSE BEFORE FIXING**: Understand the problem fully before writing code

### Process

1. **Think through the problem** - Read relevant files first
2. **Write a plan** - Create todo items in tasks/todo.md
3. **Get approval** - Check in before major changes
4. **Implement incrementally** - Mark todos complete as you go
5. **Explain changes** - High-level summary of what changed
6. **Review** - Add summary to todo.md when done

### Testing

- **423 tests** covering all modules
- Run tests after every change: `python -m pytest tests/`
- All tests must pass before committing

---

## Key Files

| File | Purpose | When to Modify |
|------|---------|----------------|
| `app_startech.py` | Streamlit UI | Adding UI features |
| `core/orchestrator.py` | Query routing | Adding new intent types |
| `handlers/*.py` | Intent handling | Changing response logic |
| `core/intent.py` | Intent classification | Adding intent patterns |
| `core/filters.py` | Filter extraction | Improving filter parsing |
| `core/search.py` | Search strategy | Changing search behavior |
| `ui/responses.py` | Response formatting | Changing output format |
| `ui/state.py` | State persistence | Adding persisted state |

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app_startech.py

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## Environment Variables

```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=startech-products
OPENAI_CHAT_MODEL=gpt-4o
```

---

## Common Tasks

### Fix a bug in search results
1. Check `handlers/search.py` for the search handler
2. Check `core/search.py` for search strategy
3. Check `core/filters.py` for filter extraction

### Add a new blocked intent
1. Add pattern to `core/intent.py`
2. Add handler to `handlers/blocked.py`
3. Register in `core/orchestrator.py`

### Change response formatting
1. Check `ui/responses.py` for formatters
2. Check `llm/response_builder.py` for LLM responses

### Debug an issue
1. Set `DEBUG_MODE = True` in `app_startech.py`
2. Check debug output in Streamlit UI
3. Check `logs/` for structured logs

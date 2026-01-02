# ST-Bot 🤖

**StarTech.com Product Chatbot** - Clean Architecture

A modern, scalable chatbot system for StarTech.com product search and recommendations.

---

## 🏗️ Architecture

```
ST-Bot/
├── app.py                  # Main Streamlit application (thin orchestration)
├── core/                   # Business logic (pure Python, testable)
│   ├── intent.py           # Intent classification
│   ├── filters.py          # Filter extraction
│   ├── search.py           # Search strategies
│   ├── connectors.py       # Connector matching
│   └── context.py          # Data models
├── llm/                    # LLM-based query understanding
│   ├── query_parser.py     # Natural language → structured data
│   ├── prompts.py          # System prompts
│   └── domain_rules.py     # Post-processing rules
├── ui/                     # UI layer (Streamlit-specific)
│   ├── responses.py        # Response formatting
│   ├── state.py            # Session state management
│   └── logging.py          # Conversation logging
├── config/                 # Configuration
│   ├── settings.py         # Environment variables
│   ├── synonyms.py         # Synonym mappings
│   └── patterns.py         # Regex patterns
├── data/                   # Data files
│   ├── categorical_values.json
│   └── sku_vocab.json
└── tests/                  # Unit & integration tests
    ├── test_intent.py
    ├── test_filters.py
    └── test_search.py
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Required
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
PINECONE_INDEX_NAME=startech-products

# Optional
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7
EMBED_MODEL=text-embedding-3-large
USE_LLM_QUERY_UNDERSTANDING=true
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=llm --cov=ui

# Run specific test file
pytest tests/test_intent.py -v
```

---

## 🎯 Design Principles

1. **Pure Functions**: Core logic is stateless and testable
2. **Dependency Injection**: Easy to mock and swap implementations
3. **Clear Data Models**: Structured state management
4. **Explicit Priority**: Intent classification has defined order
5. **Single Responsibility**: Each module has one job

---

## 📊 Features

- ✅ **Intent Classification**: Understands user intent (greeting, product search, follow-up)
- ✅ **Smart Filter Extraction**: Extracts metadata filters from natural language
- ✅ **LLM Query Understanding**: Uses GPT-4 for complex query interpretation
- ✅ **Cascading Search**: Progressive filter relaxation for best results
- ✅ **Multi-Product Responses**: Shows multiple options when appropriate
- ✅ **Context-Aware Follow-ups**: Remembers conversation history
- ✅ **Conversation Logging**: Tracks all interactions for analysis

---

## 🔧 Configuration

### Feature Flags

Toggle features via environment variables:

- `USE_LLM_QUERY_UNDERSTANDING`: Enable LLM-based query parsing (default: true)
- `ENABLE_MULTI_PRODUCT_RESPONSES`: Show multiple products (default: true)
- `LOG_CONVERSATIONS`: Save conversations to CSV (default: true)

---

## 📝 Development

### Adding New Intent Type

1. Add enum to `core/context.py`:
   ```python
   class IntentType(Enum):
       YOUR_NEW_INTENT = "your_new_intent"
   ```

2. Add classifier to `core/intent.py`:
   ```python
   def _detect_your_intent(self, prompt: str) -> bool:
       # Detection logic
   ```

3. Add to priority order in `IntentClassifier`

4. Write tests in `tests/test_intent.py`

---

## 📈 Monitoring

Conversation logs are saved to `conversations.csv` with:
- Timestamp
- Session ID
- User query
- Bot response
- Products shown
- Filters applied
- Match status

Use for:
- Success rate analysis
- Common query patterns
- Filter effectiveness
- Product recommendation quality

---

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Add tests (maintain 80%+ coverage)
4. Run `black .` to format code
5. Run `pytest` to verify tests pass
6. Submit pull request

---

## 📄 License

Internal StarTech.com use only.

---

## 🙋 Support

For questions or issues, contact the development team.

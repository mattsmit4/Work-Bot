# ST-Bot Quick Start Guide 🚀

Welcome to ST-Bot! This guide will help you get started.

---

## ✅ Current Status

**Foundation Complete! ✨**

The following are ready to use:
- ✅ Project structure
- ✅ Configuration system (settings, synonyms, patterns)
- ✅ Data models (Intent, Product, ConversationContext)
- ✅ Testing infrastructure
- ✅ Documentation

**Still To Build:**

The following modules need to be implemented:
- ⏳ `core/intent.py` - Intent classification
- ⏳ `core/filters.py` - Filter extraction
- ⏳ `core/search.py` - Search strategies
- ⏳ `core/connectors.py` - Connector matching
- ⏳ `llm/query_parser.py` - LLM query understanding
- ⏳ `ui/responses.py` - Response formatting
- ⏳ `ui/state.py` - Session state management
- ⏳ `ui/logging.py` - Conversation logging

---

## 🎯 Step-by-Step Setup

### Step 1: Extract ST-Bot
```bash
# Extract ST-Bot.tar.gz to your VS Code Repo directory
# You should now have:
#   VS Code Repo/
#     ├── Work-Bot/    (your old system - stays unchanged)
#     └── ST-Bot/      (new clean system)
```

### Step 2: Copy Data Files
```bash
cd ST-Bot
python migrate_from_old.py
```

This will copy from **Work-Bot** (your old system):
- `categorical_values.json` (product categories)
- `sku_vocab.json` (product SKUs)
- `.env` (environment variables)
- `conversations.csv` (logs - optional)

**Note:** Your Work-Bot folder is NOT modified - it stays intact!

### Step 3: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# If .env wasn't copied, create it from template
cp .env.example .env

# Edit .env and add your API keys
# PINECONE_API_KEY=...
# OPENAI_API_KEY=...
```

### Step 5: Run Tests
```bash
# Verify setup is working
pytest tests/test_sample.py -v

# Expected output: All tests passing ✅
```

### Step 6: Run App (Skeleton)
```bash
streamlit run app.py
```

You'll see a placeholder message - this is normal! We need to build the core modules next.

---

## 🏗️ Next Steps: Building Modules

We'll build modules in this order (each takes ~1-2 hours):

### Week 1: Core Logic

**Day 1-2: Intent Classification**
- Build `core/intent.py`
- Implement `IntentClassifier` class
- Add tests in `tests/test_intent.py`
- Goal: Understand what user wants (greeting, search, follow-up)

**Day 3-4: Filter Extraction**
- Build `core/filters.py`
- Implement `FilterBuilder` class
- Add tests in `tests/test_filters.py`
- Goal: Extract metadata filters from queries

**Day 5-7: Search & Connectors**
- Build `core/search.py` (cascading search)
- Build `core/connectors.py` (connector matching)
- Add tests
- Goal: Find products matching filters

### Week 2: LLM & UI

**Day 1-3: LLM Integration**
- Build `llm/query_parser.py` (port from old system)
- Build `llm/prompts.py` (system prompts)
- Build `llm/domain_rules.py` (post-processing)
- Add tests

**Day 4-7: UI Layer**
- Build `ui/responses.py` (format responses)
- Build `ui/state.py` (manage session state)
- Build `ui/logging.py` (save conversations)
- Wire everything into `app.py`

### Week 3: Integration & Testing
- Integration tests
- Compare with old system
- Fix gaps
- Deploy!

---

## 📝 Development Workflow

### Adding a New Feature

1. **Write test first:**
   ```python
   # tests/test_intent.py
   def test_new_feature():
       classifier = IntentClassifier()
       result = classifier.classify("new query")
       assert result.type == IntentType.NEW_FEATURE
   ```

2. **Run test (it fails):**
   ```bash
   pytest tests/test_intent.py::test_new_feature -v
   ```

3. **Implement feature:**
   ```python
   # core/intent.py
   def _detect_new_feature(self, prompt: str) -> bool:
       # Implementation
   ```

4. **Run test (it passes!):**
   ```bash
   pytest tests/test_intent.py::test_new_feature -v
   ```

5. **Run all tests:**
   ```bash
   pytest tests/ -v
   ```

### Code Quality Checks

```bash
# Format code
black .

# Type checking (optional)
mypy core/ llm/ ui/

# Lint code (optional)
flake8 core/ llm/ ui/
```

---

## 🧪 Testing Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=llm --cov=ui

# Run specific test file
pytest tests/test_intent.py -v

# Run specific test
pytest tests/test_intent.py::TestIntentClassifier::test_greeting -v

# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration
```

---

## 🎯 Architecture Principles to Remember

1. **Pure Functions**: Core logic should have no side effects
2. **Dependency Injection**: Pass dependencies as parameters
3. **Single Responsibility**: Each module does ONE thing
4. **Test Everything**: Aim for 80%+ coverage
5. **Clear Data Models**: Use dataclasses, not dictionaries

---

## 🆘 Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the ST-Bot directory
cd ST-Bot

# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Tests not discovering
```bash
# Make sure you have __init__.py files
touch tests/__init__.py
touch core/__init__.py

# Run from project root
cd ST-Bot
pytest
```

### Import errors
```bash
# Make sure PYTHONPATH includes current directory
export PYTHONPATH=.  # On Windows: set PYTHONPATH=.
pytest
```

---

## 📞 Next Session Prep

When ready to continue building:

1. **Say which module to work on:**
   - "Let's build core/intent.py"
   - "Let's build core/filters.py"
   - etc.

2. **I'll create the module:**
   - Write the implementation
   - Write comprehensive tests
   - Show you how to run it

3. **We'll iterate:**
   - Run tests together
   - Fix any issues
   - Move to next module

---

## 🎉 You're Ready!

The foundation is solid. Now we build the modules one by one, with tests for each.

**No more whack-a-mole. Just clean, tested, scalable code.** 💪

Questions? Just ask! Let's build this properly. 🚀

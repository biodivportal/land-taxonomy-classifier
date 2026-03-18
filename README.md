# Land Taxonomy API

A FastAPI prototype that uses an LLM to identify which CORINE Land Cover taxonomy types are described in a given text.

## How It Works

1. The API loads all land types from `taxonomy.csv` at startup.
2. A `POST /classify` request sends your text plus the full taxonomy list to an OpenAI model.
3. The LLM returns matched land types with confidence scores and reasoning.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run the server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### `GET /`
Health check. Returns the number of loaded taxonomy entries.

**Response:**
```json
{"status": "ok", "taxonomy_entries": 52}
```

---

### `GET /taxonomy`
Returns all taxonomy entries loaded from `taxonomy.csv`.

---

### `POST /classify`
Classifies text against the land taxonomy.

**Request body:**
```json
{
  "text": "Rock outcrops, unvegetated rock, cliffs, rocky surfaces",
  "top_k": 3,
  "model": "gpt-4o-mini"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | — | The text to classify |
| `top_k` | integer | No | `5` | Number of best-matching categories to return |
| `model` | string | No | `gpt-4o-mini` | OpenAI model to use |

**Response:**
```json
{
  "matches": [
    {
      "clc_code": "332",
      "english_name": "Bare Rock",
      "confidence": 0.98,
      "reason": "rock outcrops, unvegetated rock and cliffs directly described"
    },
    {
      "clc_code": "333",
      "english_name": "Sparsely Vegetated Areas",
      "confidence": 0.41,
      "reason": "rocky surfaces often co-occur with sparse pioneer vegetation"
    },
    {
      "clc_code": "331",
      "english_name": "Beaches, Dunes and Sand Plains",
      "confidence": 0.18,
      "reason": "rocky coastal cliffs may border sandy shore habitats"
    }
  ],
  "summary": "An area of exposed bare rock with cliffs and unvegetated rocky surfaces.",
  "input_text": "Rock outcrops, unvegetated rock, cliffs, rocky surfaces"
}
```

Confidence is a float from `0.0` (no match) to `1.0` (perfect match), sorted descending.

## Interactive Docs

FastAPI provides auto-generated Swagger UI at:

```
http://127.0.0.1:8000/docs
```

## Project Structure

```
land-taxonomy-api/
├── main.py            # FastAPI application
├── taxonomy.csv       # CORINE Land Cover taxonomy definitions
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── README.md
```

## Taxonomy Source

The taxonomy is based on [CORINE Land Cover (CLC)](https://land.copernicus.eu/pan-european/corine-land-cover), extended with German names and synonyms/alternative terms for improved LLM matching.

import os
import json
import asyncio
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Land Taxonomy API", version="1.0.0")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load taxonomy once at startup
_df = pd.read_csv(
    "taxonomy.csv",
    sep=";",
    encoding="utf-8-sig",
    dtype=str,
).fillna("")

TAXONOMY_ENTRIES = [
    {
        "clc_code": row["CLC Code"].strip(),
        "level": row["Level"].strip(),
        "parent_code": row["Parent Code"].strip(),
        "german_name": row["German Name"].strip(),
        "english_name": row["English Name"].strip(),
        "synonyms": row["Synonyms"].strip(),
    }
    for _, row in _df.iterrows()
    if row["English Name"].strip()
]

# Build taxonomy reference strings per level for the system prompt
def _ref_line(e: dict) -> str:
    line = f"- [CLC {e['clc_code']}] {e['english_name']}"
    if e["synonyms"]:
        line += f" (aka: {e['synonyms']})"
    return line

_l1 = [e for e in TAXONOMY_ENTRIES if e["level"] == "1"]
_l2 = [e for e in TAXONOMY_ENTRIES if e["level"] == "2"]
_l3 = [e for e in TAXONOMY_ENTRIES if e["level"] == "3"]

TAXONOMY_REFERENCE = (
    "=== Level 1 (broad categories) ===\n"
    + "\n".join(_ref_line(e) for e in _l1)
    + "\n\n=== Level 2 (sub-categories) ===\n"
    + "\n".join(_ref_line(e) for e in _l2)
    + "\n\n=== Level 3 (detailed classes) ===\n"
    + "\n".join(_ref_line(e) for e in _l3)
)

SYSTEM_PROMPT = f"""You are a land-use classification expert.
You have the following land taxonomy (CORINE Land Cover / LBM-DE based), organised in three levels:

{TAXONOMY_REFERENCE}

When given a text and a number N, identify the top N best-fitting land types.
For each match you MUST provide predictions at all three hierarchy levels:
  - level1: the matching Level-1 category (single-digit CLC code, e.g. "3")
  - level2: the matching Level-2 sub-category (two-digit CLC code, e.g. "33")
  - level3: the matching Level-3 detailed class (three-digit CLC code, e.g. "332")

Each level object has: "clc_code", "english_name", "confidence" (0.0–1.0).
Also provide a brief "reason" and an overall "confidence" for the whole match.

Return exactly the top N matches, sorted by overall confidence descending.
Respond ONLY with a valid JSON object in this exact format:
{{
  "matches": [
    {{
      "confidence": 0.95,
      "reason": "brief explanation",
      "level1": {{"clc_code": "3", "english_name": "Forest and Semi-Natural Areas", "confidence": 0.95}},
      "level2": {{"clc_code": "33", "english_name": "Open Spaces with Little or No Vegetation", "confidence": 0.92}},
      "level3": {{"clc_code": "332", "english_name": "Bare Rock", "confidence": 0.90}}
    }}
  ],
  "summary": "one sentence summary of the land described"
}}
If nothing matches at all, return matches as an empty array."""


class ClassifyRequest(BaseModel):
    text: str
    top_k: int = 5
    model: str = "gpt-4o-mini"

    @validator("top_k")
    def top_k_bounds(cls, v):
        if not 1 <= v <= 20:
            raise ValueError("top_k must be between 1 and 20")
        return v

    @validator("text")
    def text_not_too_long(cls, v):
        if len(v) > 5000:
            raise ValueError("text must be 5000 characters or fewer")
        return v


class LevelPrediction(BaseModel):
    clc_code: str
    english_name: str
    confidence: float


class TaxonomyMatch(BaseModel):
    confidence: float
    reason: str
    level1: LevelPrediction
    level2: LevelPrediction
    level3: LevelPrediction


class ClassifyResponse(BaseModel):
    matches: list[TaxonomyMatch]
    summary: str
    input_text: str


@app.get("/")
def root():
    return {"status": "ok", "taxonomy_entries": len(TAXONOMY_ENTRIES)}


@app.get("/taxonomy")
def list_taxonomy():
    return {"entries": TAXONOMY_ENTRIES}


class BatchClassifyRequest(BaseModel):
    texts: list[str]
    top_k: int = 5
    model: str = "gpt-4o-mini"

    @validator("texts")
    def texts_bounds(cls, v):
        if not 1 <= len(v) <= 20:
            raise ValueError("texts must contain between 1 and 20 items")
        for t in v:
            if len(t) > 5000:
                raise ValueError("each text must be 5000 characters or fewer")
        return v

    @validator("top_k")
    def top_k_bounds(cls, v):
        if not 1 <= v <= 20:
            raise ValueError("top_k must be between 1 and 20")
        return v


async def _classify_single(text: str, top_k: int, model: str) -> ClassifyResponse:
    if not text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Top N: {top_k}\n\nText: {text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {raw}")

    return ClassifyResponse(
        matches=data.get("matches", []),
        summary=data.get("summary", ""),
        input_text=text,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(req: ClassifyRequest):
    return await _classify_single(req.text, req.top_k, req.model)


@app.post("/classify/batch", response_model=list[ClassifyResponse])
async def classify_batch(req: BatchClassifyRequest):
    results = await asyncio.gather(
        *[_classify_single(text, req.top_k, req.model) for text in req.texts],
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            raise HTTPException(status_code=500, detail=str(r))
    return results

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

# Lookup by code for parent resolution
_by_code: dict[str, dict] = {e["clc_code"]: e for e in TAXONOMY_ENTRIES}

# Only level-3 entries go into the LLM prompt — parents are resolved automatically
_l3 = [e for e in TAXONOMY_ENTRIES if e["level"] == "3"]

TAXONOMY_REFERENCE = "\n".join(
    f"- [CLC {e['clc_code']}] {e['english_name']}"
    + (f" (aka: {e['synonyms']})" if e["synonyms"] else "")
    for e in _l3
)

SYSTEM_PROMPT = f"""You are a land-use classification expert.
You have the following detailed land taxonomy (CORINE Land Cover / LBM-DE Level-3 classes):

{TAXONOMY_REFERENCE}

When given a text and a number N, identify the top N best-fitting Level-3 classes.
Score each match with a confidence value between 0.0 and 1.0.
Return exactly the top N matches, sorted by confidence descending.
Every match MUST use a CLC code from the list above.

Respond ONLY with a valid JSON object in this exact format:
{{
  "matches": [
    {{
      "clc_code": "332",
      "english_name": "Bare Rock",
      "confidence": 0.95,
      "reason": "brief explanation"
    }}
  ],
  "summary": "one sentence summary of the land described"
}}
If nothing matches at all, return matches as an empty array."""


def _resolve_hierarchy(clc_code: str, llm_name: str, llm_confidence: float) -> dict:
    """Given a level-3 CLC code, return level1/level2/level3 sub-objects."""
    l3 = _by_code.get(clc_code, {"clc_code": clc_code, "english_name": llm_name, "parent_code": ""})
    l2_code = l3.get("parent_code", "")
    l2 = _by_code.get(l2_code, {"clc_code": l2_code, "english_name": "", "parent_code": ""})
    l1_code = l2.get("parent_code", "")
    l1 = _by_code.get(l1_code, {"clc_code": l1_code, "english_name": ""})
    return {
        "level1": {"clc_code": l1["clc_code"], "english_name": l1["english_name"], "confidence": round(llm_confidence * 0.95, 4)},
        "level2": {"clc_code": l2["clc_code"], "english_name": l2["english_name"], "confidence": round(llm_confidence * 0.97, 4)},
        "level3": {"clc_code": clc_code, "english_name": llm_name, "confidence": llm_confidence},
    }


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

    matches = []
    for m in data.get("matches", []):
        hierarchy = _resolve_hierarchy(m["clc_code"], m.get("english_name", ""), m.get("confidence", 0.0))
        matches.append({
            "confidence": m.get("confidence", 0.0),
            "reason": m.get("reason", ""),
            **hierarchy,
        })

    return ClassifyResponse(
        matches=matches,
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

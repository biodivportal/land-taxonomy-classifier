import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Land Taxonomy API", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        "german_name": row["German Name"].strip(),
        "english_name": row["English Translation"].strip(),
        "synonyms": row["Synonyms / Alternative Terms"].strip(),
    }
    for _, row in _df.iterrows()
    if row["English Translation"].strip()  # skip empty rows
]

TAXONOMY_REFERENCE = "\n".join(
    f"- [{e['clc_code'] or 'no code'}] {e['english_name']}"
    + (f" (aka: {e['synonyms']})" if e["synonyms"] else "")
    for e in TAXONOMY_ENTRIES
)

SYSTEM_PROMPT = f"""You are a land-use classification expert.
You have the following land taxonomy (CORINE Land Cover based):

{TAXONOMY_REFERENCE}

When given a text and a number N, identify the top N best-fitting taxonomy types described or implied.
Score each match with a confidence value between 0.0 and 1.0 (1.0 = perfect match, 0.0 = no match).
Return exactly the top N matches, sorted by confidence descending.
Respond ONLY with a valid JSON object in this exact format:
{{
  "matches": [
    {{
      "clc_code": "...",
      "english_name": "...",
      "confidence": 0.0,
      "reason": "brief explanation"
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


class TaxonomyMatch(BaseModel):
    clc_code: str
    english_name: str
    confidence: float
    reason: str


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


@app.post("/classify", response_model=ClassifyResponse)
def classify_text(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    response = client.chat.completions.create(
        model=req.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Top N: {req.top_k}\n\nText: {req.text}"},
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
        input_text=req.text,
    )

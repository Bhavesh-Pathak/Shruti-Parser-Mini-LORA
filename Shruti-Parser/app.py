from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq
import os
import json
import logging
import asyncio
import re
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Shruti Parser",
             description="A LangChain-based text analysis microservice")

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    intent: str
    actions: List[str]
    summary: str

# Initialize LangChain components
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    logger.info("Initializing Groq client...")
    client = Groq(api_key=api_key)
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

class GroqLLM:
    def __init__(self, client):
        self.client = client

    def _sync_call(self, prompt: str) -> str:
        """Blocking call to Groq API. Kept separate so it can be run in a thread."""
        try:
            logger.info(f"Making API call to Groq with prompt: {prompt}")
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Keep the user-requested model
                messages=[
                    {"role": "system", "content": "You are a precise text analysis assistant that provides concise, direct responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Received response from Groq: {result}")
            return result
        except Exception as e:
            # Provide clearer error for model-not-found without changing original exception semantics
            msg = str(e)
            logger.error(f"Error in Groq API call: {msg}")
            if "model_not_found" in msg or "does not exist" in msg or "model `" in msg:
                raise RuntimeError(f"Groq model not available or inaccessible: {msg}") from e
            raise

    async def invoke(self, prompt: str) -> str:
        """Async wrapper that runs the blocking Groq call in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self._sync_call, prompt)

llm = GroqLLM(client)

# Classification prompt template
classification_template = """Classify this text into exactly one of these categories: command, query, teaching, data.

Text: {text}

Output format: Return ONLY one of these exact words: command, query, teaching, data
No other text, punctuation, or explanation allowed."""

async def classify_text(text: str) -> str:
    try:
        # Build and send classification prompt
        prompt = classification_template.format(text=text)
        raw = (await llm.invoke(prompt)).strip().lower()

        # Normalize model output to a single word token (letters only)
        tokens = re.findall(r"[a-zA-Z]+", raw)
        allowed = {"command", "query", "teaching", "data"}
        if tokens:
            intent = tokens[0].lower()
            if intent in allowed:
                logger.info(f"Classification result: {intent} (raw: {raw})")
                return intent

        # Fallback: give a stricter instruction asking for exactly one of the allowed words
        logger.warning(f"Unexpected classification output: {raw}. Running fallback clarification prompt.")
        fallback_prompt = (
            "CLASSIFY_ONLY: Return ONLY one exact word from [command, query, teaching, data].\n"
            f"Text: {text}\n\nReturn exactly one word with no punctuation."
        )
        raw2 = (await llm.invoke(fallback_prompt)).strip().lower()
        tokens2 = re.findall(r"[a-zA-Z]+", raw2)
        if tokens2 and tokens2[0].lower() in allowed:
            intent2 = tokens2[0].lower()
            logger.info(f"Fallback classification result: {intent2} (raw: {raw2})")
            return intent2

        # If still not valid, default to 'query' and log full details for debugging
        logger.error(f"Classification failed to return a valid intent. raw1='{raw}', raw2='{raw2}'")
        return "query"
    except Exception as e:
        logger.error(f"Error in classification: {str(e)}")
        raise

# Extraction prompt template — enforce JSON array of full action phrases
extraction_template = """Extract actionable instructional phrases from the input text.
Action phrases should be short, self-contained instructions or recommendations (preserve objects and qualifiers).
Return the result as a JSON array of strings ONLY. No extra text or explanation.

Examples:
Input: "When writing unit tests, prefer small, focused test cases. Use fixtures for shared setup and assert one behavior per test."
Output: ["prefer small, focused test cases", "use fixtures for shared setup", "assert one behavior per test"]

Now extract from this text:
{text}
"""

async def extract_actions_from_text(text: str) -> List[str]:
    """Extract action phrases. First try strict JSON parsing, then fall back to heuristic parsing that
    prefers multi-word phrases and instruction verbs.
    """
    if not text or not text.strip():
        return []

    prompt = extraction_template.format(text=text)
    try:
        raw = await llm.invoke(prompt)
        logger.info(f"Raw extraction response: {raw}")

        # Try strict JSON parse first
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = []
                for a in parsed:
                    if not isinstance(a, str):
                        continue
                    s = re.sub(r"\s+", " ", a.strip()).rstrip('.')
                    if s:
                        cleaned.append(s)
                if cleaned:
                    # deduplicate while preserving order
                    seen = set()
                    dedup = []
                    for s in cleaned:
                        key = s.lower()
                        if key not in seen:
                            seen.add(key)
                            dedup.append(s)
                    return dedup
        except json.JSONDecodeError:
            logger.debug("Extraction response not valid JSON, falling back to heuristic parsing")

        # Fallback heuristics: split on newlines or bullets first, then commas; prefer multi-word phrases
        candidates = re.split(r'[\n;•\u2022\-]\s*', raw)
        if len(candidates) == 1:
            candidates = re.split(r',\s*', raw)

        actions = []
        for seg in candidates:
            seg_clean = seg.strip().strip('"').strip("'").rstrip('.').strip()
            if not seg_clean:
                continue
            # Accept segments with >=2 words or those containing instruction verbs
            if len(seg_clean.split()) >= 2 or re.search(r'\b(prefer|use|create|install|run|configure|migrate|restart|email|prepare|finalize|assert|clone|deploy)\b', seg_clean, re.I):
                actions.append(re.sub(r"\s+", " ", seg_clean))

        # deduplicate while preserving order
        seen = set()
        dedup = []
        for a in actions:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(a)
        return dedup

    except Exception as e:
        logger.error(f"Error extracting actions: {str(e)}")
        return []

# Summarization prompt template
summarization_template = """Create a one-sentence summary of this text.
Rules:
1. Must be factual and based only on the input text
2. No meta-commentary or references to missing information
3. Keep it concise and direct

Text: {text}

Summary:"""


# Combined analyze prompt - request a strict JSON object with the three required fields
analyze_combined_template = """You must return a single valid JSON object and nothing else. The object must have these keys:
1) "intent": one of the exact strings: "command", "query", "teaching", "data"
2) "actions": a JSON array of short instructional phrases (strings). If there are no actions, return an empty array.
3) "summary": a single-sentence factual summary of the input text.

Example output:
{"intent":"teaching","actions":["prefer small, focused test cases","use fixtures for shared setup"],"summary":"When writing unit tests, prefer small focused test cases and use fixtures for shared setup."}

Now produce only the JSON object for the following input text (no explanation, no backticks):
{text}
"""

async def summarize_text_content(text: str) -> str:
    try:
        prompt = summarization_template.format(text=text)
        result = await llm.invoke(prompt)
        logger.info(f"Summarization result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise


async def analyze_combined(text: str) -> AnalysisResponse:
    """Try a single-shot JSON analyze call; if it fails, fall back to the pipeline that calls individual prompts."""
    try:
        prompt = analyze_combined_template.format(text=text)
        raw = await llm.invoke(prompt)
        logger.info(f"Combined analyze raw response: {raw}")

        try:
            obj = json.loads(raw)
            # validate shape
            intent = obj.get("intent")
            actions = obj.get("actions")
            summary = obj.get("summary")
            allowed = {"command", "query", "teaching", "data"}
            if (isinstance(intent, str) and intent in allowed
                    and isinstance(actions, list)
                    and all(isinstance(a, str) for a in actions)
                    and isinstance(summary, str)):
                # normalize and return
                return AnalysisResponse(intent=intent, actions=[a.strip() for a in actions], summary=summary.strip())
            else:
                logger.warning("Combined analyze returned JSON but failed validation, falling back to pipeline")
        except Exception:
            logger.warning("Combined analyze response not valid JSON, falling back to pipeline")

        # Fallback: run the pipeline (classification, extraction, summarization)
        intent = (await classify_text(text)).strip().lower()
        actions = await extract_actions_from_text(text)
        summary = (await summarize_text_content(text)).strip()
        return AnalysisResponse(intent=intent, actions=actions, summary=summary)

    except Exception as e:
        logger.error(f"Error in combined analyze: {str(e)}")
        # On failure, bubble as HTTP error in caller
        raise

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(text_input: TextInput):
    try:
        if not text_input.text or not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty")
        # Get intent classification
        intent = (await classify_text(text_input.text)).strip().lower()
        
        # Extract action phrases
        actions = await extract_actions_from_text(text_input.text)
        
        # Get summary
        summary = (await summarize_text_content(text_input.text)).strip()
        
        return AnalysisResponse(
            intent=intent,
            actions=actions,
            summary=summary
        )
    except Exception as e:
        # If the handler already raised an HTTPException (like our 400 input validation), re-raise it
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_actions(text_input: TextInput):
    try:
        if not text_input.text or not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty")
        actions = await extract_actions_from_text(text_input.text)
        return {"actions": actions}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error in /extract endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(text_input: TextInput):
    try:
        if not text_input.text or not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty")
        summary = (await summarize_text_content(text_input.text)).strip()
        return {"summary": summary}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error in /summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI

from videorag.config.settings import get_settings
from videorag.input.select_video import pick_video_ids

ReasoningEffort = Literal["minimal", "low", "medium", "high"]
Verbosity = Literal["low", "medium", "high"]

SYSTEM_PROMPT = """
You are helping build evaluation data for a single-video educational retrieval system.

You will receive one lecture video transcript.

Your job is to first understand the transcript at a high level:
- what the lecture is broadly about
- what the main ideas are
- what a learner would likely want to understand from it

Then generate exactly 5 learner-style queries for that one video.

The goal is not to ask about tiny transcript details.
The goal is to generate realistic learner questions that reflect the main topics, ideas, concepts, or explanations in the video.

Rules:
- Use only the transcript provided.
- Generate exactly 5 queries.
- Each query must be answerable by the transcript.
- Each query must be a single coherent question.
- Do not combine multiple distinct questions into one.
- Avoid questions joined by "and" when they are really asking two separate things.
- Prefer natural learner questions over keyword-style queries.
- Prefer questions about the main ideas, core concepts, or major explanations in the video.
- Do not overfocus on narrow sub-parts unless they are clearly central to the lecture.
- Write the queries as if the learner has not watched the video yet and is trying to understand what it can teach them.
- Do not assume the learner already knows the specific examples, sections, or terminology used later in the transcript.
- Do not copy transcript wording too closely.
- Avoid near-paraphrases of transcript sentences.
- Prefer concept-focused, explanation-seeking, interpretation-seeking, comparison-seeking, or understanding-oriented learner questions.
- Favor questions that ask to understand an idea rather than asking the system to directly perform a task or solve a calculation.
- If the transcript includes examples, use them to motivate broader understanding-oriented questions about the idea being taught.
- Do not invent information not supported by the transcript.
- Do not mention timestamps.
- Do not mention that the answer comes from a transcript.
- Vary the wording and style of the queries a little.

Output valid JSON only in this format:
{
  "video_id": "...",
  "queries": [
    {
      "query_id": 1,
      "query": "..."
    },
    {
      "query_id": 2,
      "query": "..."
    },
    {
      "query_id": 3,
      "query": "..."
    },
    {
      "query_id": 4,
      "query": "..."
    },
    {
      "query_id": 5,
      "query": "..."
    }
  ]
}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain valid JSON.")
        return json.loads(text[start : end + 1])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty transcript file: {path}")
    return text


def build_user_payload(*, video_id: str, transcript: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "transcript": transcript,
    }


def ask_llm_for_queries(
    *,
    user_payload: dict[str, Any],
    model: str,
    reasoning_effort: ReasoningEffort,
    verbosity: Verbosity,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
            },
        ],
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
    )

    return extract_json(response.output_text)


def get_registry_video_entry(registry: dict[str, Any], video_id: str) -> dict[str, Any]:
    videos = registry.get("videos", {})
    if video_id not in videos:
        raise KeyError(f"Video '{video_id}' not found in registry.")

    entry = videos[video_id]
    if not isinstance(entry, dict):
        raise ValueError(f"Registry entry for '{video_id}' is invalid.")

    return entry


def normalize_query_result(result: dict[str, Any], video_id: str) -> dict[str, Any]:
    queries = result.get("queries", [])
    if not isinstance(queries, list):
        queries = []

    normalized_queries = []
    for idx, item in enumerate(queries[:5], start=1):
        if not isinstance(item, dict):
            continue

        query_text = str(item.get("query", "")).strip()
        if not query_text:
            continue

        normalized_queries.append(
            {
                "query_id": idx,
                "query": query_text,
            }
        )

    return {
        "video_id": result.get("video_id", video_id),
        "queries": normalized_queries,
    }


def main() -> None:
    settings = get_settings()

    registry_path = settings.paths.registry_path
    derived_dir = settings.paths.derived_dir

    registry = load_json(registry_path)

    video_ids = pick_video_ids("Select videos to generate learner queries for")

    model_name = "gpt-5-mini"
    reasoning_effort = cast(ReasoningEffort, "medium")
    verbosity = cast(Verbosity, "low")

    output_dir = derived_dir / "eval_queries"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Generating queries for {video_id} ===")

        entry = get_registry_video_entry(registry, video_id)

        transcript_path = derived_dir / video_id / "transcript_raw.txt"
        transcript_text = read_text(transcript_path)

        input_payload = build_user_payload(
            video_id=video_id,
            transcript=transcript_text,
        )

        raw_result = ask_llm_for_queries(
            user_payload=input_payload,
            model=model_name,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

        generated_queries = normalize_query_result(raw_result, video_id)

        combined_output = {
            "video_id": video_id,
            "group_name": entry.get("group_name"),
            "source_local_filename": entry.get("source_local_filename"),
            "input_payload": input_payload,
            "generated_queries": generated_queries["queries"],
        }

        output_path = output_dir / f"{video_id}_queries.json"
        output_path.write_text(
            json.dumps(combined_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote queries to: {output_path}")


if __name__ == "__main__":
    main()

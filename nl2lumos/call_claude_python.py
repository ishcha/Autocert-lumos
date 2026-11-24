#!/usr/bin/env python3
"""
Call Bedrock Claude with claude_prompt_py.jinja for each spec in nlspecs/,
write generated Python scripts to py_specs_claude/.
With an optional CLI spec file, run only that file and store its result.
Run nl2lumos/compare_loc.py to compare LOC between py_specs_claude/ and spec_progs_claude/.
"""

import argparse
import json
from pathlib import Path

import boto3
from jinja2 import Environment, FileSystemLoader

import os
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def render_jinja_file(template_path: Path, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    return template.render(**context)


def create_bedrock_client():
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def call_claude(client, model_id: str, prompt: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    if "content" in payload:
        return "".join(
            block["text"] for block in payload["content"] if block["type"] == "text"
        )
    return json.dumps(payload, indent=2)


def main():
    project_root = Path(__file__).resolve().parents[1]
    nl2lumos = project_root / "nl2lumos"
    nlspecs_dir = nl2lumos / "nlspecs"

    parser = argparse.ArgumentParser(description="Call Bedrock Claude for NL → Python.")
    parser.add_argument(
        "spec_file",
        nargs="?",
        help="Optional: single spec Jinja file to run; if omitted, run all specs in nlspecs/",
    )
    args = parser.parse_args()

    template_path = nl2lumos / "claude_prompt_py.jinja"
    py_out_dir = nl2lumos / "py_specs_claude"
    py_out_dir.mkdir(parents=True, exist_ok=True)

    if args.spec_file is not None:
        spec_files = [Path(args.spec_file)]
    else:
        spec_files = sorted(nlspecs_dir.glob("*.jinja"))

    if not spec_files:
        print("No .jinja spec files found in nl2lumos/nlspecs/ (or the given path is empty).")
        return

    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        raise SystemExit("Set AWS_BEARER_TOKEN_BEDROCK to your Bedrock API key first, "
                         "e.g. export AWS_BEARER_TOKEN_BEDROCK=...")
    client = create_bedrock_client()

    for spec_path in spec_files:
        stem = spec_path.stem
        spec_text = render_jinja_file(spec_path, {})
        prompt = render_jinja_file(template_path, {"specification": spec_text})
        output = call_claude(client, CLAUDE_MODEL_ID, prompt)
        out_path = py_out_dir / f"{stem}.py"
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

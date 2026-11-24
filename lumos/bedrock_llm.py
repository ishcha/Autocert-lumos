"""Amazon Bedrock-backed LLM provider for NL->Lumos translation."""

from __future__ import annotations

import json
from typing import Any

import boto3


class BedrockLLMProvider:
    """Bedrock Claude-compatible callable with Lumos provider interface."""

    def __init__(
        self,
        model: str,
        *,
        region: str = "us-east-1",
        gen_length: int = 768,
        temperature: float | None = 0.2,
        top_p: float | None = None,
        thinking_budget_tokens: int | None = None,
        system_message: str | None = None,
        anthropic_version: str = "bedrock-2023-05-31",
        profile_name: str | None = None,
    ):
        self.model = model
        self.region = region
        self.gen_length = gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget_tokens = thinking_budget_tokens
        self.system_message = system_message
        self.anthropic_version = anthropic_version
        self.profile_name = profile_name

        self._client = None
        # Shared callable across trial copies.
        self.__lumos_shared_callable__ = True

    def start(self):
        session = boto3.Session(profile_name=self.profile_name) if self.profile_name else boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=self.region)

    def stop(self):
        self._client = None

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        parts = []
        for block in payload.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _has_thinking_only(payload: dict[str, Any]) -> bool:
        content = payload.get("content", []) or []
        if not isinstance(content, list) or not content:
            return False
        saw_thinking = False
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                return False
            if block_type == "thinking":
                saw_thinking = True
        return saw_thinking

    def _make_body(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "anthropic_version": self.anthropic_version,
            "max_tokens": int(self.gen_length),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        # Extended thinking mode for Claude models on Bedrock.
        # When thinking is enabled, do not send sampling modifiers.
        if self.thinking_budget_tokens is not None:
            budget = int(self.thinking_budget_tokens)
            if budget < 1024:
                raise ValueError("thinking_budget_tokens must be >= 1024 for Bedrock Claude")
            if budget >= int(self.gen_length):
                raise ValueError("thinking_budget_tokens must be less than max_tokens")
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        else:
            # Some Bedrock models reject requests that include both temperature
            # and top_p. Prefer temperature when both are configured.
            if self.temperature is not None:
                body["temperature"] = float(self.temperature)
            elif self.top_p is not None:
                body["top_p"] = float(self.top_p)
        if self.system_message:
            body["system"] = self.system_message
        return body

    def __call__(self, prompt: str) -> str:
        if self._client is None:
            self.start()

        try:
            response = self._client.invoke_model(
                modelId=self.model,
                body=json.dumps(self._make_body(prompt)),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            text = self._extract_text(payload)
            if text:
                return text

            stop_reason = payload.get("stop_reason")
            if self._has_thinking_only(payload) and stop_reason == "max_tokens":
                raise RuntimeError(
                    "Bedrock returned only thinking blocks and hit max_tokens; "
                    "increase translator max tokens or reduce/disable thinking budget."
                )

            raise RuntimeError(
                "Bedrock returned no text content for translator output "
                f"(stop_reason={stop_reason!r})."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock request failed for model '{self.model}' in region '{self.region}': {exc}"
            ) from exc

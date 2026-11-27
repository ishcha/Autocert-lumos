"""vLLM-backed LLM provider for Lumos."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import requests
from openai import OpenAI

from .local_model import strip_thinking


_DEFAULT_VLLM_FLAGS = {
    "max_logprobs": -1,
    "return_tokens_as_token_ids": True,
    "logprobs_mode": "processed_logprobs",
    "trust_remote_code": True,
    "max_model_len": 4092,
    "gpu_memory_utilization": 0.5,
    "enable_prefix_caching": True,
    "max_num_seqs": 256,
}


def _find_vllm_binary():
    import shutil
    local = Path(sys.executable).parent / "vllm"
    return str(local) if local.exists() else (shutil.which("vllm") or "vllm")


def _build_vllm_command(model, port, flags):
    cmd = [_find_vllm_binary(), "serve", model, "--port", str(port)]
    for key, val in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif val is not None:
            cmd.extend([flag, str(val)])
    return cmd


class LLMProvider:
    """vLLM-backed LLM callable for Lumos."""

    def __init__(
        self,
        model,
        server_addr=None,
        port=8081,
        gen_length=256,
        temperature=1.0,
        top_p=0.99,
        top_k=None,
        use_chat_template=True,
        system_message=None,
        gpu_devices=None,
        api_key=None,
        vllm_flags=None,
        thinking=False,
    ):
        self.model = model
        self.server_addr = server_addr
        self.port = port
        self.gen_length = gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_chat_template = use_chat_template
        self.system_message = system_message
        self.gpu_devices = gpu_devices
        # Models that think first (Qwen3, Qwen3.5) answer directly unless thinking=True, as in LocalModel.
        self.thinking = thinking
        # Overrides of _DEFAULT_VLLM_FLAGS, e.g. {"gpu_memory_utilization": 0.3}.
        self.vllm_flags = {**_DEFAULT_VLLM_FLAGS, **(vllm_flags or {})}
        # Priority: explicit arg, LUMOS_API_KEY, OPENAI_API_KEY.
        self.api_key = api_key or os.environ.get("LUMOS_API_KEY") or os.environ.get("OPENAI_API_KEY")

        self._proc = None
        self._client = None
        self._model_name = model
        self._log_fh = None
        self._server_log = Path("lumos_vllm_server.log")
        # Stateful network client/server handle; share across trials intentionally.
        self.__lumos_shared_callable__ = True

    def start(self):
        timeout = 30
        if self.server_addr is None:
            self.server_addr = f"http://localhost:{self.port}"
            self._start_server()
            timeout = 600
        else:
            self.server_addr = self._normalize_server_addr(self.server_addr)

        self._wait_for_server_ready(timeout=timeout)
        self._client = self._create_client()

        # Resolve model ID exposed by server.
        try:
            served = [m.id for m in self._client.models.list().data]
        except Exception:
            # If listing models fails, keep requested model id and let request
            # path surface provider-specific errors.
            return

        if not served:
            return
        if self._model_name in served:
            return

        # Local/self-hosted vLLM often exposes exactly one model id that may
        # differ slightly from user input. Keep this compatibility fallback.
        if len(served) == 1:
            self._model_name = served[0]
            return

        shown = ", ".join(served[:10])
        more = " ..." if len(served) > 10 else ""
        raise RuntimeError(
            f"Requested model '{self._model_name}' is not available at {self.server_addr}/v1. "
            f"Available models: {shown}{more}"
        )

    def _normalize_server_addr(self, server_addr):
        addr = server_addr.rstrip("/")
        if addr.endswith("/v1"):
            addr = addr[:-3]
        return addr.rstrip("/")

    def _auth_headers(self):
        token = self.api_key or os.environ.get("LUMOS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not token or token == "dummy":
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _wait_for_server_ready(self, timeout=30):
        started = time.time()
        deadline = started + timeout
        addr = self.server_addr.rstrip("/")
        ipv4 = addr.replace("://localhost", "://127.0.0.1", 1)
        candidates = [addr] if ipv4 == addr else [addr, ipv4]
        headers = self._auth_headers()
        last_error = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                rc = self._proc.returncode
                self.stop()
                raise RuntimeError(
                    f"vLLM server exited with code {rc}. "
                    f"Check log: {self._server_log.resolve()}"
                )
            for candidate in candidates:
                try:
                    response = requests.get(f"{candidate}/health", timeout=5, headers=headers)
                    if response.status_code == 200:
                        self.server_addr = candidate
                        if self._proc is not None:
                            print(f"[lumos] Server healthy ({time.time() - started:.1f}s)")
                        return
                    # Some managed OpenAI-compatible gateways deny /health. Treat
                    # /v1/models reachability as a sufficient readiness signal.
                    models_response = requests.get(
                        f"{candidate}/v1/models", timeout=5, headers=headers
                    )
                    if models_response.status_code in (200, 401, 403):
                        self.server_addr = candidate
                        return
                    last_error = RuntimeError(
                        f"Health status {response.status_code}; models status {models_response.status_code}"
                    )
                except requests.exceptions.RequestException as exc:
                    last_error = exc
            time.sleep(1)

        if self._proc is not None:
            self.stop()
            raise RuntimeError(
                f"vLLM server failed to start within {timeout}s. "
                f"Check log: {self._server_log.resolve()}"
            )

        tried = ", ".join(candidates)
        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        raise RuntimeError(
            f"Unable to reach vLLM server within {timeout}s. Tried {tried}. Last error: {detail}"
        )

    def _create_client(self):
        api_key = self.api_key or os.environ.get("LUMOS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
        return OpenAI(
            base_url=f"{self.server_addr}/v1",
            api_key=api_key,
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0),
        )

    @staticmethod
    def _extract_text_from_content(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)

        if isinstance(content, list):
            parts = []
            for item in content:
                text = None
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text = item["text"]
                    elif isinstance(item.get("content"), str):
                        text = item["content"]
                    elif isinstance(item.get("output_text"), str):
                        text = item["output_text"]
                    elif isinstance(item.get("text"), dict):
                        value = item["text"].get("value")
                        if isinstance(value, str):
                            text = value
                else:
                    text = getattr(item, "text", None)
                    if not isinstance(text, str):
                        out_text = getattr(item, "output_text", None)
                        if isinstance(out_text, str):
                            text = out_text
                        else:
                            value = getattr(out_text, "value", None)
                            if isinstance(value, str):
                                text = value
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)

        model_dump = getattr(content, "model_dump", None)
        if callable(model_dump):
            return LLMProvider._extract_text_from_content(model_dump())

        return str(content)

    def _extract_chat_message_text(self, response):
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return str(getattr(choice, "text", "") or "")

        text = self._extract_text_from_content(getattr(message, "content", None))
        if text != "":
            return text

        # Compatibility fallback for providers that emit text/refusal outside `content`.
        for field in ("refusal", "reasoning_content", "reasoning", "text"):
            value = getattr(message, field, None)
            fallback = self._extract_text_from_content(value)
            if fallback != "":
                return fallback

        return str(getattr(choice, "text", "") or "")

    def _start_server(self):
        cmd = _build_vllm_command(self.model, self.port, self.vllm_flags)
        env = os.environ.copy()
        if self.gpu_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_devices)

        print(f"[lumos] Starting vLLM: {self.model} on port {self.port}")
        print(f"[lumos]   Command: {' '.join(cmd)}")
        self._log_fh = open(self._server_log, "w")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def stop(self):
        if self._proc is None:
            return
        print("[lumos] Stopping server ...")
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self._proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        self._proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        print("[lumos] Server stopped")

    def _generate_once(self, prompt):
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            if self.system_message:
                messages.insert(0, {"role": "system", "content": self.system_message})
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=self.gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body=self._extra_body(chat=True),
            )
            return strip_thinking(self._extract_chat_message_text(response))

        response = self._client.completions.create(
            model=self._model_name,
            prompt=prompt,
            max_tokens=self.gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=self._extra_body(chat=False),
        )
        return str(getattr(response.choices[0], "text", "") or "")

    def _extra_body(self, chat):
        """vLLM's extensions to the OpenAI request: top_k, and the thinking switch for chat templates."""
        body = {}
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if chat:
            # read by Qwen3-style chat templates and ignored by others
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        return body

    def __call__(self, prompt):
        for attempt in range(3):
            try:
                return self._generate_once(prompt)
            except Exception as exc:
                if attempt == 2:
                    msg = str(exc)
                    if msg.strip().lower() == "connection error.":
                        msg = (
                            f"Connection error while calling {self.server_addr}/v1. "
                            f"Ensure vLLM is reachable and try --server-addr http://127.0.0.1:{self.port}."
                        )
                    else:
                        msg = (
                            f"LLM request failed for model '{self._model_name}' "
                            f"at {self.server_addr}/v1: {msg}"
                        )
                    raise RuntimeError(msg) from exc
                time.sleep(2 * (attempt + 1))
                self._client = self._create_client()

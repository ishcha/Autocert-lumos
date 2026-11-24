"""A local HuggingFace model, run in-process with transformers, to bind as L.

    from lumos.local_model import LocalModel
    L = LocalModel("Qwen/Qwen2.5-0.5B-Instruct")      # downloads on first use
    L("Hello")                                         # -> the model's reply, a string

No server is needed, so this also works in a notebook (e.g. Google Colab).
Chat models get the prompt as one user message through their chat template;
other models continue the prompt text. Models that think before answering
(e.g. Qwen3 / Qwen3.5) answer directly unless thinking=True, and any
<think>...</think> block is removed from the reply.
"""

import re
from typing import Optional

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def dtype_kwargs(on_gpu: bool = True) -> dict:
    """The from_pretrained dtype argument for this machine.

    float16 on GPUs without native bfloat16 (compute capability < 8, e.g.
    Colab's T4), otherwise the model's own dtype. transformers 5 renamed
    torch_dtype to dtype; Colab may still have 4.x.
    """
    import torch
    import transformers

    value = "auto"
    if on_gpu and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        value = torch.float16
    name = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    return {name: value}


def strip_thinking(text: str) -> str:
    """The final answer: <think> blocks removed; nothing if thinking never finished."""
    text = _THINKING.sub("", text)
    if "<think>" in text:  # ran out of tokens while thinking
        text = text.split("<think>", 1)[0]
    return text.strip()


class LocalModel:
    """prompt -> reply, generated with transformers."""

    def __init__(self, model: str, *, max_new_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 0.99, top_k: Optional[int] = None, system_message: Optional[str] = None,
                 thinking: bool = False, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == -1:
            device = "cpu"
        elif isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            device = f"cuda:{device}"
        placement = {"device_map": "auto"} if device == "auto" else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **dtype_kwargs(device != "cpu"), **placement)
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_message = system_message
        self.thinking = thinking
        self.chat = getattr(self.tokenizer, "chat_template", None) is not None

    def __call__(self, prompt: str) -> str:
        import torch

        if self.chat:
            messages = [{"role": "user", "content": str(prompt)}]
            if self.system_message:
                messages.insert(0, {"role": "system", "content": self.system_message})
            # enable_thinking is read by Qwen3-style templates and ignored by others.
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=self.thinking)
        else:
            text = str(prompt)
        inputs = self.tokenizer([text], return_tensors="pt", add_special_tokens=not self.chat).to(self.model.device)
        options = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            options.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
            if self.top_k is not None:
                options["top_k"] = self.top_k
        else:
            options["do_sample"] = False
        pad = self.tokenizer.pad_token_id
        options["pad_token_id"] = self.tokenizer.eos_token_id if pad is None else pad
        with torch.no_grad():
            output = self.model.generate(**inputs, **options)
        reply = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return strip_thinking(reply)

    def __repr__(self) -> str:
        return f"LocalModel({self.name!r})"

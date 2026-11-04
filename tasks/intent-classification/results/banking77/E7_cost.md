[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Rows 3080, tokenizer mlx-community/Qwen3.5-4B-OptiQ-4bit.
Prompt tokens: mean 24.6, p95 40. Completion tokens: mean 8.9.

| model | $ per 1k requests | basis | source |
|---|---|---|---|
| Opus 5 (with caching) | 2.4200 | $2.42 per 1k requests | https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/ |
| Jev 1.13 | 0.1100 | $0.11 per 1k requests | https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/ |
| OpenAI Decisions gpt-6-luna | 0.0025 | $0.10 per 1M uncached input tokens x our mean 24.6 prompt tokens | https://developers.openai.com/api/docs/guides/decisions |
| Small model on a Mac (mlx-community/Qwen3.5-4B-OptiQ-4bit) | not metered | electricity not counted | measured here |

The Decisions row uses our tokenizer's count; OpenAI's own tokenizer and request wrapper will differ.

Cascade cost (Conformal Cascade, Theorem 4): c_small + c_frontier x (1 - acceptance), acceptance 0.58.
Frontier dollars avoided per 1k requests (estimate; assumes the small model's cost is zero):

| frontier | always-frontier $/1k | cascade frontier $/1k | avoided $/1k |
|---|---|---|---|
| Opus 5 (with caching) | 2.4200 | 1.0164 | 1.4036 |
| Jev 1.13 | 0.1100 | 0.0462 | 0.0638 |
| OpenAI Decisions gpt-6-luna | 0.0025 | 0.0010 | 0.0014 |

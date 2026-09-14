"""
AI layer.

Business code depends on :class:`app.ai.base.AIProvider` - never on a vendor SDK.
Implemented providers: YandexGPT (Yandex Cloud / AI Studio flavours), any
OpenAI-compatible endpoint, Ollama (local), and ``DisabledProvider`` so the whole
application keeps working when no AI is configured.
"""

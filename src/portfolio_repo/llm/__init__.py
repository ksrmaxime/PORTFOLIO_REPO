from portfolio_repo.llm.client import LocalLLMClient, LLMConfig

_default_client: LocalLLMClient | None = None


def get_local_llm() -> LocalLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LocalLLMClient(LLMConfig())
    return _default_client

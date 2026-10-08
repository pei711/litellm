from typing import Final

from litellm.types.decisions import OpenAIDecisionUsage


def test_openai_decision_usage_omitted_details_do_not_leak_cache_counts() -> None:
    first: Final = OpenAIDecisionUsage(input_tokens=10, output_tokens=1, total_tokens=11)
    second: Final = OpenAIDecisionUsage(input_tokens=20, output_tokens=2, total_tokens=22)

    object.__setattr__(first.input_tokens_details, "cached_tokens", 99)
    object.__setattr__(first.output_tokens_details, "reasoning_tokens", 7)

    assert second.input_tokens_details.cached_tokens == 0
    assert second.output_tokens_details.reasoning_tokens == 0

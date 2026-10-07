use std::collections::BTreeMap;

use litellm_cost::{
    PromptConvention, Usage,
    catalog::{self, EstimateRequest},
};
use litellm_model_catalog::PricingCatalog;

fn count(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn counts(attributes: &BTreeMap<String, String>, keys: &[&str]) -> Option<Option<u64>> {
    let values = keys
        .iter()
        .filter_map(|key| attributes.get(*key))
        .map(|value| count(value))
        .collect::<Option<Vec<_>>>()?;
    let first = values.first().copied();
    values
        .iter()
        .all(|value| Some(*value) == first)
        .then_some(first)
}

fn text<'a>(attributes: &'a BTreeMap<String, String>, keys: &[&str]) -> Option<Option<&'a str>> {
    let values: Vec<_> = keys
        .iter()
        .filter_map(|key| attributes.get(*key))
        .map(String::as_str)
        .collect();
    let first = values.first().copied();
    values
        .iter()
        .all(|value| Some(*value) == first)
        .then_some(first)
}

pub fn estimate_cost(
    attributes: &BTreeMap<String, String>,
    catalog: &PricingCatalog,
) -> Option<f64> {
    if attributes.iter().any(|(key, value)| {
        ["audio", "image", "video"]
            .iter()
            .any(|kind| key.contains(kind))
            && value != "0"
    }) {
        return None;
    }
    let model = attributes
        .get("gen_ai.response.model")
        .or_else(|| attributes.get("gen_ai.request.model"))?;
    if model.is_empty() {
        return None;
    }
    let input = counts(
        attributes,
        &["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"],
    )??;
    let output = counts(
        attributes,
        &[
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
        ],
    )??;
    let read = counts(attributes, &["gen_ai.usage.cache_read.input_tokens"])?.unwrap_or_default();
    let write = counts(
        attributes,
        &[
            "gen_ai.usage.cache_write.input_tokens",
            "gen_ai.usage.cache_creation.input_tokens",
        ],
    )?
    .unwrap_or_default();
    let five = counts(
        attributes,
        &["anthropic.usage.cache_creation.ephemeral_5m_input_tokens"],
    )?;
    let one = counts(
        attributes,
        &["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"],
    )?;
    let reasoning = counts(attributes, &["gen_ai.usage.reasoning.output_tokens"])?;
    let response_tier = text(
        attributes,
        &[
            "openai.response.service_tier",
            "anthropic.response.service_tier",
            "gen_ai.openai.response.service_tier",
        ],
    )?;
    let request_tier = text(
        attributes,
        &[
            "openai.request.service_tier",
            "gen_ai.openai.request.service_tier",
        ],
    )?;
    let start = match attributes.get("litellm.trace.start_ns") {
        Some(value) => {
            count(value.strip_prefix('-').unwrap_or(value))?;
            Some(value.parse::<i64>().ok()?)
        }
        None => None,
    };
    let matched = catalog.resolve(
        model,
        attributes.get("gen_ai.provider.name").map(String::as_str),
        attributes.get("gen_ai.system").map(String::as_str),
    )?;
    catalog::estimate(
        matched.fields,
        &EstimateRequest {
            usage: Usage {
                prompt_tokens: input,
                completion_tokens: output,
                cache_read_tokens: read,
                cache_write_tokens: write,
                cache_write_5m_tokens: five,
                cache_write_1h_tokens: one,
                prompt_convention: PromptConvention::IncludesCache,
            },
            service_tier: response_tier.or(request_tier),
            reasoning_tokens: reasoning,
            billed_at_ns: start,
        },
    )
}

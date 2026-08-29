# liberated-claude

A gateway that serves the bootstrap URL, model discovery, and Messages API that
Claude Desktop expects, routing to whatever providers an XML file names.

It exists because the alternatives clamp every model to a 200000-token context
window. A 1M-context model routed through one of those is a 200k model. This one
advertises the window the model actually has.

## What it gets right

**Context windows are reported truthfully.** Claude Desktop decides whether to
offer a model's 1M-context variant from the discovery response:

```js
l(model.supports_1m, model.max_input_tokens)
  where l = (a, b) => typeof a == "boolean" ? a : typeof b == "number" && b >= 1e6
```

Each model's configured `contextWindow` is sent as `max_input_tokens`, so a
1310720-token model is offered at 1310720.

**Models are not silently dropped.** Desktop screens every discovered model ID
and discards anything matching a non-Anthropic token - `deepseek`, `glm`, `gpt`,
`grok`, `qwen`, `gemini` and about thirty more. A model survives only if its ID
passes that screen or it carries a recognized `anthropic_family_tier`. Upstream
IDs are hex-encoded behind a `claude-lc-` prefix so no rejection token can
survive, and every model requires a tier in config.

**Caching is preserved rather than stripped.** Providers on `cache="explicit"`
receive Anthropic `cache_control` blocks untouched. Providers on
`cache="implicit"` get a byte-stable prefix and no directive, which is what
automatic prefix caching needs. Either way the cache counters come back as
`cache_read_input_tokens` and `cache_creation_input_tokens`, with cached tokens
subtracted out of `input_tokens` per Anthropic's accounting.

**Costs are detected from upstream.** Rates are fetched from the provider's own
catalog at startup and published as `inferenceModelPricing`, so the Usage page
shows real numbers instead of estimating everything at Anthropic list price.

## Setup

```sh
go-toolchain                      # build and test; binary lands in build/
cp config.example.xml config.xml  # then edit
./build/liberated-claude -config config.xml
```

Point Claude Desktop's **Bootstrap config URL** at `/bootstrap`, for example
`http://127.0.0.1:8787/bootstrap`. Everything else - gateway URL, key, model
list, pricing - comes from that response.

`${VAR}` in the config is read from the environment at load. An unset variable
is a startup error rather than an empty key, because an empty key upstream
produces an opaque 401.

## Endpoints

| Path            | Purpose                                              |
| --------------- | ---------------------------------------------------- |
| `/bootstrap`    | Config overlay Claude Desktop fetches at launch      |
| `/v1/models`    | Model discovery, with real `max_input_tokens`        |
| `/v1/messages`  | Anthropic Messages API, proxied and translated       |
| `/healthz`      | Liveness, unauthenticated                            |

## Running under pm2

```sh
pm2 start ecosystem.config.js
```

`watch` targets `build/liberated-claude`, so rebuilding restarts the gateway.

## Cache modes

| Mode       | Behavior                                                          |
| ---------- | ----------------------------------------------------------------- |
| `explicit` | `cache_control` blocks forwarded untouched                        |
| `implicit` | No directive sent; provider caches prefixes; hits reported back   |
| `none`     | `cache_control` stripped so strict providers do not reject it     |

Set on a `<provider>` and optionally overridden per `<model>`.

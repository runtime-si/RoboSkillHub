"""Tests for the vlm tool bundle — per-provider request shaping, all mocked.

No network, no GPU: the openrouter provider (the default) is exercised
through ``httpx.MockTransport``, and the vertex provider behind an import
guard (skipped when google-genai is absent).
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from gap_core.errors import ToolError
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def vlm():
    """The vlm bundle's tools module.

    The vlm bundle serves out-of-process (``serving.protocol:
    stdio-msgpack``), so ``load_skills`` does not import its ``tools.py``
    in-process — import it directly for these in-process unit tests.
    """
    import importlib.util

    tools_path = ROOT / "tools" / "vlm" / "tools.py"
    spec = importlib.util.spec_from_file_location("vlm_tools_under_test", tools_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # the default env file is under ~, so a developer's real keys are never read
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in (
        "GAP_VLM_PROVIDER",
        "GAP_VLM_MODEL",
        "GAP_VLM_BASE_URL",
        "GAP_VLM_API_KEY",
        "GAP_VLM_PROJECT_ID",
        "GAP_VLM_REGION",
        "GAP_LLM_PROVIDER",
        "GAP_LLM_MODEL",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "GAP_VLM_ENV_FILE",
        "GAP_VLM_ASTRA_ENV",
        "GAP_VLM_GEMINI_ENV",
        "ASTRA_API_KEY",
        "ASTRA_URL",
        "ASTRA_AUTH",
        "ASTRA_MODEL",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
    ):
        monkeypatch.delenv(var, raising=False)


def _mock_openrouter(vlm, monkeypatch, reply: str):
    """Install an httpx.MockTransport on the vlm bundle's http seam."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    return captured


# ---------------------------------------------------------------------------
# openrouter provider (default) — OpenAI-compatible chat completions
# ---------------------------------------------------------------------------


def test_openrouter_is_default_with_data_url_image(vlm, image, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="What is on the table?", image=image)
    assert out == {"text": "a red mug"}

    # Default provider targets OpenRouter with the bundle-default model.
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-router"
    payload = captured["payload"]
    assert payload["model"] == vlm.DEFAULT_MODEL == "gemini-3.1-flash-lite-preview"
    assert payload["max_tokens"] == vlm._MAX_TOKENS == 4096
    assert payload["temperature"] == 0.0

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is on the table?"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")),
        image,
    )


def test_max_tokens_env_override(vlm, image, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
    captured = _mock_openrouter(vlm, monkeypatch, reply="ok")
    monkeypatch.setenv("GAP_VLM_MAX_TOKENS", "8192")
    vlm.query(prompt="q", image=image)
    assert captured["payload"]["max_tokens"] == 8192
    monkeypatch.setenv("GAP_VLM_MAX_TOKENS", "not-a-number")
    vlm.query(prompt="q", image=image)
    assert captured["payload"]["max_tokens"] == vlm._MAX_TOKENS


def test_openrouter_model_env_override_and_multiple_images(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_MODEL", "anthropic/claude-sonnet-4")
    captured = _mock_openrouter(vlm, monkeypatch, reply="compared")

    second = np.zeros((2, 3, 3), dtype=np.uint8)
    vlm.query(prompt="compare", image=image, images=[second])

    payload = captured["payload"]
    assert payload["model"] == "anthropic/claude-sonnet-4"
    content = payload["messages"][0]["content"]
    # Text first, then one image_url block per image.
    assert [b["type"] for b in content] == ["text", "image_url", "image_url"]


def test_openrouter_custom_base_url(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://vlm.test/v1")
    monkeypatch.setenv("GAP_VLM_API_KEY", "sk-test")
    monkeypatch.setenv("GAP_VLM_MODEL", "gcp/google/gemini-3-flash-preview")
    captured = _mock_openrouter(vlm, monkeypatch, reply="two cups")

    out = vlm.query(prompt="What objects are on the table?", image=image)
    assert out == {"text": "two cups"}

    assert captured["url"] == "http://vlm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["payload"]["model"] == "gcp/google/gemini-3-flash-preview"


def test_openrouter_text_only_query(vlm, monkeypatch):
    captured = _mock_openrouter(vlm, monkeypatch, reply="hi there")
    vlm.query(prompt="hello")
    content = captured["payload"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hello"}]


def test_openrouter_backend_failure_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_MODEL", "m")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)

    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"error": "overloaded"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert len(attempts) == 3


# ---------------------------------------------------------------------------
# vertex provider (Gemini only; import-guarded)
# ---------------------------------------------------------------------------


def test_vertex_rejects_claude_model(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")
    with pytest.raises(ToolError, match="Gemini models only"):
        vlm.query(prompt="hi")


def test_vertex_provider_routes_gemini_models_to_genai(vlm, image, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(text="gemini says hi")

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi", image=image)
    assert out == {"text": "gemini says hi"}
    assert captured["client_kwargs"] == {
        "vertexai": True,
        "project": "test-project",
        "location": "global",
    }
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["contents"][0] == "hi"
    # Deterministic decoding, as the dev servicer's production path; the cap
    # is raised from its 1024 so a thinking model's answer is not cut (#20).
    assert captured["config"].temperature == 0.0
    assert captured["config"].max_output_tokens == 4096


def test_vertex_gemini_retries_transient_failures(vlm, image, monkeypatch):
    """The vertex Gemini path retries like the openrouter path (3 attempts)."""
    pytest.importorskip("google.genai")
    from google import genai

    attempts: list[int] = []

    class _FlakyClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 transient")
            return SimpleNamespace(text="third time lucky")

    monkeypatch.setattr(genai, "Client", _FlakyClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi")
    assert out == {"text": "third time lucky"}
    assert len(attempts) == 3


def test_vertex_gemini_exhausted_retries_raise_tool_error(vlm, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    class _DeadClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            raise RuntimeError("permanently overloaded")

    monkeypatch.setattr(genai, "Client", _DeadClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="hi")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_provider_kwarg_overrides_env(vlm, monkeypatch):
    # Env says vertex (and is unconfigured, so it would fail) — the kwarg wins.
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    captured = _mock_openrouter(vlm, monkeypatch, reply="a red mug")

    out = vlm.query(prompt="q", provider="openrouter")
    assert out == {"text": "a red mug"}
    assert captured["payload"]["messages"][0]["content"][0]["text"] == "q"


def test_unknown_provider_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "bedrock")
    with pytest.raises(ToolError, match="unknown provider 'bedrock'"):
        vlm.query(prompt="q")


def test_invalid_image_rejected(vlm):
    bad = np.zeros((4, 6), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        vlm.query(prompt="q", image=bad)


# ---------------------------------------------------------------------------
# Yes/no coercion (first standalone yes/no word; legacy substring fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", True),
        ("yes.", True),
        ("YES — clearly visible.", True),
        ("The answer is yes", True),
        ("Yes, the object matches the description.", True),
        ("Eyes on the table", True),  # legacy substring fallback quirk
        ("No", False),
        ("no.", False),
        ("Absolutely not", False),
        ("I cannot tell", False),
        ("", False),
        # First standalone word wins — the legacy substring check would
        # mislabel both of these (the G1 verify-gate failure mode):
        ("No, although the label literally says YES on it.", False),
        ("NO. The item appears to be a small book.", False),
    ],
)
def test_query_yes_no_coercion(vlm, monkeypatch, text, expected):
    _mock_openrouter(vlm, monkeypatch, reply=text)
    out = vlm.query_yes_no(prompt="Is the sauce in the basket?")
    assert out == {"answer": expected, "text": text}


def test_query_yes_no_appends_explicit_instruction(vlm, monkeypatch):
    """query_yes_no must elicit a parseable YES/NO-first reply.

    Without the instruction (and at temperature > 0) models answer
    affirmatively in prose with no literal "yes" — which the coercion
    mislabels as False. A false "No" from the perceiving-objects verify
    gate rejects a correct exterior pick and forces the degraded
    single-view wrist fallback.
    """
    captured = _mock_openrouter(vlm, monkeypatch, reply="Yes.")

    vlm.query_yes_no(prompt="Is this a cream cheese box?")

    (text_block,) = captured["payload"]["messages"][0]["content"]
    assert text_block["type"] == "text"
    assert text_block["text"].startswith("Is this a cream cheese box?")
    assert "YES or NO first" in text_block["text"]


def _mock_azure(vlm, monkeypatch, reply: str):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("api-key")
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": reply}]},
                ]
            },
        )

    monkeypatch.setattr(
        vlm, "_http_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    return captured


def test_azure_responses_api_without_temperature(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "forskillhub")
    monkeypatch.setenv("GAP_LLM_MODEL", "gemini-3-flash-preview")
    captured = _mock_azure(vlm, monkeypatch, reply="a pen over a hole")

    out = vlm.query(prompt="what is shown?", image=image)

    assert out == {"text": "a pen over a hole"}
    assert captured["url"] == "https://res.services.ai.azure.com/openai/v1/responses"
    assert captured["api_key"] == "az-key"
    payload = captured["payload"]
    assert payload["model"] == "forskillhub"
    assert "temperature" not in payload and "max_tokens" not in payload
    assert payload["max_output_tokens"] == 4096
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "what is shown?"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_azure_yes_no_and_missing_config(vlm, image, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "forskillhub")
    _mock_azure(vlm, monkeypatch, reply="Yes, the pen is over the hole.")
    assert (
        vlm.query_yes_no(prompt="over the hole?", image=image, provider="azure")["answer"] is True
    )

    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT")
    with pytest.raises(ToolError, match="AZURE_OPENAI_DEPLOYMENT"):
        vlm.query(prompt="x", image=image, provider="azure")


# ---------------------------------------------------------------------------
# astra provider — the Responses API behind an Azure API Management gateway
# ---------------------------------------------------------------------------


def test_astra_reads_its_env_file_and_sends_the_key_its_way(vlm, image, monkeypatch, tmp_path):
    env = tmp_path / "openai.env"
    env.write_text(
        "export OPENAI_API_KEY='astra-key'\n"
        "ASTRA_URL=https://gw.example.net/astra/openai/v1/responses?api-version=x\n"
        "ASTRA_AUTH=azure\n"
        "ASTRA_MODEL=astra-large  # the gateway's model\n"
    )
    monkeypatch.setenv("GAP_VLM_ASTRA_ENV", str(env))
    captured = _mock_azure(vlm, monkeypatch, reply="six rings")

    out = vlm.query(prompt="how many rings?", image=image, provider="astra")

    assert out == {"text": "six rings"}
    assert captured["url"] == "https://gw.example.net/astra/openai/v1/responses"
    assert captured["api_key"] == "astra-key"
    assert captured["auth"] is None
    assert captured["payload"]["model"] == "astra-large"


def test_astra_bearer_auth_sends_only_a_bearer_token(vlm, image, monkeypatch, tmp_path):
    env = tmp_path / "vlm.env"
    env.write_text(
        "ASTRA_API_KEY=astra-key\nASTRA_URL=https://gw.example.net/v1\nASTRA_AUTH=bearer\n"
    )
    monkeypatch.setenv("GAP_VLM_ENV_FILE", str(env))
    captured = _mock_azure(vlm, monkeypatch, reply="ok")

    vlm.query(prompt="x", image=image, provider="astra", model="m")

    assert captured["url"] == "https://gw.example.net/v1/responses"
    assert captured["auth"] == "Bearer astra-key"
    assert captured["api_key"] is None


def test_astra_prefers_astra_api_key_and_never_reads_openai_key_from_the_environment(
    vlm, image, monkeypatch, tmp_path
):
    env = tmp_path / "vlm.env"
    env.write_text("ASTRA_URL=https://gw.example.net/v1\nASTRA_MODEL=m\n")
    monkeypatch.setenv("GAP_VLM_ENV_FILE", str(env))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-for-api-openai-com")

    with pytest.raises(ToolError, match="ASTRA_API_KEY") as raised:
        vlm.query(prompt="x", image=image, provider="astra")
    assert "sk-for-api-openai-com" not in str(raised.value)

    env.write_text(env.read_text() + "OPENAI_API_KEY=file-key\n")
    monkeypatch.setenv("ASTRA_API_KEY", "env-astra-key")
    captured = _mock_azure(vlm, monkeypatch, reply="ok")
    vlm.query(prompt="x", image=image, provider="astra")
    assert captured["api_key"] == "env-astra-key"


def test_astra_without_its_env_file_names_what_it_needs_and_no_secret(
    vlm, image, monkeypatch, tmp_path
):
    env = tmp_path / "openai.env"
    env.write_text("OPENAI_API_KEY=astra-key\nASTRA_URL=http://plain.example.net\nASTRA_MODEL=m\n")
    monkeypatch.setenv("GAP_VLM_ASTRA_ENV", str(env))

    with pytest.raises(ToolError, match="https ASTRA_URL") as raised:
        vlm.query(prompt="x", image=image, provider="astra")
    assert "astra-key" not in str(raised.value)


def test_a_failing_gateway_error_carries_no_key(vlm, image, monkeypatch, tmp_path):
    env = tmp_path / "vlm.env"
    env.write_text("ASTRA_API_KEY=astra-secret\nASTRA_URL=https://gw.example.net/v1\n")
    monkeypatch.setenv("GAP_VLM_ENV_FILE", str(env))
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "no"}))
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))

    with pytest.raises(ToolError, match="unavailable after 3 attempts") as raised:
        vlm.query(prompt="x", image=image, provider="astra", model="m")
    assert "astra-secret" not in str(raised.value)


# ---------------------------------------------------------------------------
# gemini provider — Google's OpenAI-compatible endpoint
# ---------------------------------------------------------------------------


def test_gemini_takes_the_first_key_of_its_env_file(vlm, image, monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEYS="g-first, g-second"\n')
    monkeypatch.setenv("GAP_VLM_GEMINI_ENV", str(env))
    captured = _mock_openrouter(vlm, monkeypatch, reply="a blue ring on top")

    out = vlm.query(prompt="top ring?", image=image, provider="gemini", model="gemini-x")

    assert out == {"text": "a blue ring on top"}
    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert captured["auth"] == "Bearer g-first"
    assert captured["payload"]["model"] == "gemini-x"


def test_gemini_without_a_key_says_so(vlm, image, monkeypatch, tmp_path):
    monkeypatch.setenv("GAP_VLM_GEMINI_ENV", str(tmp_path / "absent.env"))
    with pytest.raises(ToolError, match="GEMINI_API_KEY"):
        vlm.query(prompt="x", image=image, provider="gemini")


def test_gemini_reads_the_default_env_file_under_home(vlm, image, monkeypatch, tmp_path):
    default = tmp_path / "home" / ".config" / "viper" / "vlm.env"
    default.parent.mkdir(parents=True)
    default.write_text("export GEMINI_API_KEYS=home-key\n")
    assert Path(vlm.DEFAULT_ENV_FILE).expanduser() == default
    captured = _mock_openrouter(vlm, monkeypatch, reply="ok")

    vlm.query(prompt="x", image=image, provider="gemini")

    assert captured["auth"] == "Bearer home-key"
    assert captured["payload"]["model"] == vlm.DEFAULT_MODEL


def test_environment_key_wins_and_a_per_provider_file_wins_over_the_shared_one(
    vlm, image, monkeypatch, tmp_path
):
    shared, own = tmp_path / "shared.env", tmp_path / "gemini.env"
    shared.write_text("GEMINI_API_KEY=shared-key\n")
    own.write_text("GEMINI_API_KEY=own-key\n")
    monkeypatch.setenv("GAP_VLM_ENV_FILE", str(shared))
    captured = _mock_openrouter(vlm, monkeypatch, reply="ok")

    vlm.query(prompt="x", image=image, provider="gemini")
    assert captured["auth"] == "Bearer shared-key"

    monkeypatch.setenv("GAP_VLM_GEMINI_ENV", str(own))
    vlm.query(prompt="x", image=image, provider="gemini")
    assert captured["auth"] == "Bearer own-key"

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    vlm.query(prompt="x", image=image, provider="gemini")
    assert captured["auth"] == "Bearer env-key"


# ---------------------------------------------------------------------------
# env files, and nothing private in the bundle
# ---------------------------------------------------------------------------


def test_env_file_reads_only_the_named_single_word_values(vlm, tmp_path):
    env = tmp_path / "vlm.env"
    env.write_text(
        "# a comment\n"
        "\n"
        "export A='quoted value'\n"
        'B="x"  # trailing comment\n'
        "C=two words\n"
        "D='unbalanced\n"
        "OTHER=not-asked-for\n"
        "not a line\n"
    )

    assert vlm._env_file(str(env), {"A", "B", "C", "D"}) == {"A": "quoted value", "B": "x"}
    assert vlm._env_file(str(tmp_path / "absent.env"), {"A"}) == {}
    assert vlm._env_file(str(tmp_path), {"A"}) == {}


def test_the_bundle_hard_codes_no_key_and_no_private_path():
    bundle = ROOT / "tools" / "vlm"
    private = re.compile(
        r"/Users/|/home/|/root/|simgen|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9_-]{20,}"
    )
    for file in [*bundle.glob("*.py"), bundle / "SKILL.md"]:
        found = private.findall(file.read_text())
        assert not found, f"{file.name} holds {found}"

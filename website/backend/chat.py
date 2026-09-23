import logging
import os

import anthropic

log = logging.getLogger("chat")

MODEL = os.environ.get("CHAT_MODEL", "claude-opus-5")
MAX_TOKENS = 1500
MAX_HISTORY = 10
MAX_MESSAGE_CHARS = 1000

SYSTEM_PROMPT = """You are the help assistant inside "Indian Currency Authenticator", a website that \
uses two AI models to check a photo of an Indian banknote: stage 1 decides whether the image is a \
banknote at all, and stage 2 estimates whether the note is genuine or counterfeit.

You help users with two things:
1. Indian banknote security features and how to check a note by hand (watermark, security thread, \
see-through register, latent image, micro-lettering, intaglio print feel, colour-shifting ink, and so on), \
plus what to do if they suspect a counterfeit (do not pass it on, hand it to a bank branch or the police).
2. Understanding the website's results. Grade A-D reflects authenticity (A best for a genuine note, D for a \
confident counterfeit). Risk level is the chance the note is bad. Rating is 1-5 stars of authenticity. \
Decision strength (BORDERLINE, MODERATE, STRONG) says how far the model's score is from the 50% line. \
Image quality (GOOD, FAIR, POOR) says whether the photo was good enough; a POOR photo should be retaken.

Rules:
- The website's result is a screening aid from a machine-learning model, not proof. It can be wrong, \
especially on blurry, dark or unusual photos. For any real doubt, tell the user to have the note checked \
at a bank.
- Only state security features you are confident about. If unsure of a detail for a specific denomination \
or series, say so and point to the Reserve Bank of India's official guidance instead of guessing.
- Stay on topic. If asked about something unrelated, politely say you can only help with banknotes and this \
website.
- Be concise: short paragraphs or a short list, plain language, no long preambles."""


class ChatDisabled(Exception):
    pass


class ChatRateLimited(Exception):
    pass


class ChatUnavailable(Exception):
    pass


_client = None


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=30.0, max_retries=1)
    return _client


def ask(messages: list[dict]) -> str:
    if not enabled():
        raise ChatDisabled

    kwargs = {}
    if not MODEL.startswith("claude-haiku"):
        kwargs["output_config"] = {"effort": "low"}

    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages[-MAX_HISTORY:],
            **kwargs,
        )
    except anthropic.RateLimitError:
        raise ChatRateLimited
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        log.error("Chat misconfigured (%s): %s", type(e).__name__, e)
        raise ChatDisabled
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        log.error("Chat upstream error (%s): %s", type(e).__name__, e)
        raise ChatUnavailable

    if response.stop_reason == "refusal":
        return "Sorry, I can't help with that request. Ask me about banknote security features or your scan result."

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ChatUnavailable
    return text

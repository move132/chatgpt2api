from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote, urlparse

import requests
from services.config import config
from utils.log import logger


TELEGRAM_API_BASE = "https://api.telegram.org"
TelegramImagePayload = str | bytes


def get_telegram_api_url(bot_token: str, method: str) -> str:
    return f"{TELEGRAM_API_BASE}/bot{bot_token}/{method}"

def _normalize_image_inputs(image_inputs: Iterable[str | bytes | bytearray | memoryview]) -> list[TelegramImagePayload]:
    normalized: list[TelegramImagePayload] = []
    for item in image_inputs:
        if isinstance(item, str):
            candidate = item.strip()
            if candidate:
                normalized.append(candidate)
            continue
        if isinstance(item, (bytes, bytearray, memoryview)):
            candidate = bytes(item)
            if candidate:
                normalized.append(candidate)
    return normalized


def _try_send_images_to_tg(data: list[dict[str, Any]], prompt: str) -> None:
    if not data:
        return

    image_inputs: list[TelegramImagePayload] = []
    for item in data:
        b64_json = str(item.get("b64_json") or "")
        if b64_json:
            try:
                image_inputs.append(base64.b64decode(b64_json))
            except Exception:
                pass
            continue

        url = str(item.get("url") or "")
        if url:
            local_image = _load_local_image_from_url(url)
            image_inputs.append(local_image if local_image is not None else url)

    if not image_inputs:
        return

    try:
        _dispatch_images_to_tg(image_inputs, prompt)
    except Exception as exc:
        logger.warning({"event": "tg_image_send_fail", "error": str(exc)})


def _dispatch_images_to_tg(image_inputs: list[TelegramImagePayload], prompt: str) -> None:
    thread = threading.Thread(
        target=_send_images_to_tg_worker,
        args=(image_inputs, prompt),
        name="tg-image-sender",
        daemon=True,
    )
    thread.start()


def _send_images_to_tg_worker(image_inputs: list[TelegramImagePayload], prompt: str) -> None:
    try:
        send_images_to_telegram(image_inputs, prompt_content=prompt)
    except Exception as exc:
        logger.warning({"event": "tg_image_send_fail", "error": str(exc)})


def _load_local_image_from_url(url: str) -> bytes | None:
    candidate = str(url or "").strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    path = unquote(parsed.path or "")
    marker = "/images/"
    marker_index = path.find(marker)
    if marker_index < 0:
        return None

    relative = path[marker_index + len(marker):].lstrip("/")
    if not relative:
        return None

    file_path = (config.images_dir / relative).resolve()
    images_root = config.images_dir.resolve()
    try:
        file_path.relative_to(images_root)
    except ValueError:
        return None
    if not file_path.is_file():
        return None
    try:
        return file_path.read_bytes()
    except Exception:
        return None


def iter_image_outputs_with_tg(outputs: Iterable[Any], send_to_tg: bool = True) -> Iterator[Any]:
    data: list[dict[str, Any]] = []
    prompt = ""
    message = ""
    try:
        for output in outputs:
            kind = str(getattr(output, "kind", "") or "")
            if kind == "message":
                text = str(getattr(output, "text", "") or "").strip()
                if text and not message:
                    message = text
            elif kind == "result":
                output_data = getattr(output, "data", None)
                if isinstance(output_data, list):
                    data.extend(item for item in output_data if isinstance(item, dict))
                    for item in output_data:
                        if not isinstance(item, dict):
                            continue
                        revised_prompt = str(item.get("revised_prompt") or "")
                        if revised_prompt and not prompt:
                            prompt = revised_prompt
            yield output
    finally:
        if send_to_tg:
            _try_send_images_to_tg(data, prompt or message)


def _ensure_telegram_ok(response: requests.Response) -> None:
    payload = None
    try:
        payload = response.json()
    except Exception:
        payload = None

    if response.status_code >= 400:
        if isinstance(payload, dict):
            description = str(payload.get("description") or "").strip()
            error_code = payload.get("error_code") or response.status_code
            if description:
                raise RuntimeError(f"Telegram API {error_code}: {description}")
        body = (response.text or "").strip()
        if body:
            raise RuntimeError(f"Telegram API HTTP {response.status_code}: {body[:500]}")
        response.raise_for_status()

    if isinstance(payload, dict) and payload.get("ok") is False:
        description = str(payload.get("description") or "").strip()
        error_code = payload.get("error_code") or response.status_code
        raise RuntimeError(f"Telegram API {error_code}: {description or 'request failed'}")


def get_telegram_chat_id_from_update(update: dict) -> str:
    chat_id = (
        update.get("message", {}).get("chat", {}).get("id")
        or update.get("edited_message", {}).get("chat", {}).get("id")
        or update.get("channel_post", {}).get("chat", {}).get("id")
        or update.get("edited_channel_post", {}).get("chat", {}).get("id")
        or update.get("callback_query", {}).get("message", {}).get("chat", {}).get("id")
    )
    return "" if chat_id is None else str(chat_id)


def resolve_telegram_chat_id(bot_token: str, chat_id: str | None = None) -> str:
    if chat_id:
        return str(chat_id)

    response = requests.get(
        get_telegram_api_url(bot_token, "getUpdates"),
        timeout=30,
    )
    _ensure_telegram_ok(response)
    payload = response.json()
    updates = payload.get("result")

    if not isinstance(updates, list) or not updates:
        raise RuntimeError(
            "Telegram chat_id not found. Send a message to the bot first or set TELEGRAM_CHAT_ID."
        )

    for update in reversed(updates):
        resolved_chat_id = get_telegram_chat_id_from_update(update)
        if resolved_chat_id:
            return resolved_chat_id

    raise RuntimeError("Telegram getUpdates returned data, but no usable chat_id was found.")


def chunk_list(items: list[TelegramImagePayload], size: int) -> list[list[TelegramImagePayload]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _load_demo_images(image_inputs: Iterable[str]) -> list[TelegramImagePayload]:
    demo_images: list[TelegramImagePayload] = []
    for item in image_inputs:
        candidate = str(item or "").strip()
        if not candidate:
            continue

        file_path = Path(candidate).expanduser()
        if file_path.is_file():
            demo_images.append(file_path.read_bytes())
            continue

        demo_images.append(candidate)

    return demo_images


def _send_single_photo(bot_token: str, chat_id: str, image: TelegramImagePayload, caption: str) -> None:
    if isinstance(image, str):
        response = requests.post(
            get_telegram_api_url(bot_token, "sendPhoto"),
            json={
                "chat_id": chat_id,
                "photo": image,
                "caption": caption,
            },
            timeout=120,
        )
        _ensure_telegram_ok(response)
        return

    response = requests.post(
        get_telegram_api_url(bot_token, "sendPhoto"),
        data={
            "chat_id": chat_id,
            "caption": caption,
        },
        files={
            "photo": ("image.png", image, "image/png"),
        },
        timeout=120,
    )
    _ensure_telegram_ok(response)


def _send_media_group(bot_token: str, chat_id: str, images: list[TelegramImagePayload], caption: str) -> None:
    for group_index, group in enumerate(chunk_list(images, 10)):
        media: list[dict[str, object]] = []
        files: dict[str, tuple[str, bytes, str]] = {}

        for index, image in enumerate(group):
            item: dict[str, object] = {"type": "photo"}
            if isinstance(image, str):
                item["media"] = image
            else:
                attach_name = f"file{index}"
                item["media"] = f"attach://{attach_name}"
                files[attach_name] = (f"image_{group_index + 1}_{index + 1}.png", image, "image/png")
            if group_index == 0 and index == 0:
                item["caption"] = caption
            media.append(item)

        if files:
            response = requests.post(
                get_telegram_api_url(bot_token, "sendMediaGroup"),
                data={
                    "chat_id": chat_id,
                    "media": json.dumps(media, ensure_ascii=False),
                },
                files=files,
                timeout=120,
            )
        else:
            response = requests.post(
                get_telegram_api_url(bot_token, "sendMediaGroup"),
                json={
                    "chat_id": chat_id,
                    "media": media,
                },
                timeout=120,
            )
        _ensure_telegram_ok(response)


def send_images_to_telegram(
    image_urls: Iterable[str | bytes | bytearray | memoryview],
    prompt_content: str = "",
) -> None:
    images = _normalize_image_inputs(image_urls)
    if not images:
        return

    telegram_bot_token = config.bot_token
    if not telegram_bot_token:
        return

    resolved_chat_id = resolve_telegram_chat_id(
        telegram_bot_token,
        config.chat_id,
    )
    caption = (prompt_content or "Generated image").strip()[:900]

    if len(images) == 1:
        _send_single_photo(telegram_bot_token, resolved_chat_id, images[0], caption)
        return

    _send_media_group(telegram_bot_token, resolved_chat_id, images, caption)


__all__ = [
    "get_telegram_api_url",
    "get_telegram_chat_id_from_update",
    "_try_send_images_to_tg",
    "iter_image_outputs_with_tg",
    "resolve_telegram_chat_id",
    "send_images_to_telegram",
]

# uv run python -m utils.tg_image_sender
if __name__ == "__main__":
    demo_caption = "Telegram image sender demo"
    demo_image_inputs = [
        str(Path(__file__).resolve().parents[1] / "assets" / "222.png"),
    ]
    demo_images = _load_demo_images(demo_image_inputs)
    if not demo_images:
        raise RuntimeError("No demo images found.")

    send_images_to_telegram(
        demo_images,
        prompt_content=demo_caption,
    )
    print(f"Sent {len(demo_images)} image(s) to Telegram successfully.")

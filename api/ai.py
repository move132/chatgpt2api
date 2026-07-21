from __future__ import annotations

import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_shape, request_text
from services.editable_file_task_service import editable_file_task_service
from services.log_service import LoggedCall
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
    openai_search,
)
from utils.helper import has_response_image_generation_tool, is_image_chat_request
from utils.log import logger


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=4)
    size: str | None = None
    quality: str = "auto"
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


class SearchRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class EditableFileTaskRequest(BaseModel):
    prompt: str = ""
    base64_images: list[str] = Field(default_factory=list)
    client_task_id: str | None = None


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def _request_headers(request: Request) -> dict[str, str | list[str]]:
    headers: dict[str, str | list[str]] = {}
    for raw_name, raw_value in request.headers.raw:
        name = raw_name.decode("latin-1")
        value = raw_value.decode("latin-1")
        existing = headers.get(name)
        if existing is None:
            headers[name] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            headers[name] = [existing, value]
    return headers


def _append_parameter(parameters: dict[str, object], name: str, value: object) -> None:
    existing = parameters.get(name)
    if existing is None:
        parameters[name] = value
    elif isinstance(existing, list):
        existing.append(value)
    else:
        parameters[name] = [existing, value]


def _summarize_image_parameter(value: object, field_name: str = "") -> object:
    if isinstance(value, dict):
        return {key: _summarize_image_parameter(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [_summarize_image_parameter(item, field_name) for item in value]
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered_field = field_name.lower()
    is_image_field = lowered_field in {
        "image", "image[]", "images", "images[]", "mask", "mask[]", "b64_json", "base64", "data",
    }
    if stripped.startswith("data:image/"):
        header = stripped.split(",", 1)[0]
        return {"type": "data_url", "media_type": header.split(";", 1)[0], "length": len(stripped)}
    if is_image_field and not stripped.lower().startswith(("http://", "https://")):
        return {"type": "inline_image_data", "length": len(stripped)}
    return value


async def _json_request_parameters(request: Request, fallback: object) -> object:
    try:
        return await request.json()
    except Exception:
        return fallback


async def _image_edit_request_parameters(request: Request) -> tuple[object, list[dict[str, object]]]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        parameters = await _json_request_parameters(request, {})
        return _summarize_image_parameter(parameters), []

    form = await request.form()
    parameters: dict[str, object] = {}
    files: list[dict[str, object]] = []
    for name, value in form.multi_items():
        if isinstance(value, UploadFile):
            files.append({
                "field": name,
                "filename": value.filename or "",
                "content_type": value.content_type or "",
                "size": value.size,
            })
            continue
        _append_parameter(parameters, name, _summarize_image_parameter(value, name))
    return parameters, files


def _log_image_request(
        request: Request,
        parameters: object,
        files: list[dict[str, object]] | None = None,
) -> None:

    client = getattr(request, "client", None)
    log_payload = {
        "event": "image_generation_incoming_request",
        "method": request.method,
        "path": request.url.path,
        "client_host": getattr(client, "host", "") if client else "",
        "client_port": getattr(client, "port", None) if client else None,
        "query_params": list(request.query_params.multi_items()),
        "headers": _request_headers(request),
        "parameters": parameters,
    }
    if files:
        log_payload["files"] = files
    # 这里按排障需求记录原始请求头（包括 Authorization 等敏感字段）。
    # 预先序列化为字符串，避免 Logger 对 token 类字段自动打码。
    logger.info(json.dumps(log_payload, ensure_ascii=False, default=str))


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        parameters = await _json_request_parameters(request, body.model_dump(mode="python"))
        _log_image_request(request, parameters)
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_v1_image_generations.handle, payload)

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        parameters, files = await _image_edit_request_parameters(request)
        _log_image_request(request, parameters, files)
        payload, image_sources, mask_sources = await parse_image_edit_request(request)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        payload["images"] = await read_image_sources(image_sources)
        if mask_sources:
            payload["mask"] = await read_image_sources(mask_sources)
        payload["base_url"] = resolve_image_base_url(request)
        return await call.run(openai_v1_image_edit.handle, payload)

    @router.post("/v1/chat/completions")
    async def create_chat_completion(
            body: ChatCompletionRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        payload = body.model_dump(mode="python")
        if is_image_chat_request(payload):
            parameters = await _json_request_parameters(request, payload)
            _log_image_request(request, parameters)
        identity = require_identity(authorization)
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(
            identity,
            "/v1/chat/completions",
            model,
            "文本生成",
            request_text=request_preview,
            request_shape=request_shape(payload.get("messages")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/responses")
    async def create_response(
            body: ResponseCreateRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        payload = body.model_dump(mode="python")
        if has_response_image_generation_tool(payload):
            parameters = await _json_request_parameters(request, payload)
            _log_image_request(request, parameters)
        identity = require_identity(authorization)
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(
            identity,
            "/v1/responses",
            model,
            "Responses",
            request_text=request_preview,
            request_shape=request_shape(payload.get("input")),
        )
        await filter_or_log(call, request_preview)
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    @router.post("/v1/search")
    async def search(body: SearchRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        call = LoggedCall(identity, "/v1/search", openai_search.MODEL, "搜索", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        return await call.run(openai_search.handle, body.model_dump(mode="python"))

    @router.get("/v1/editable-file-tasks")
    async def list_editable_file_tasks(ids: str = "", authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        task_ids = [item.strip() for item in ids.split(",") if item.strip()]
        return await run_in_threadpool(editable_file_task_service.list_tasks, identity, task_ids)

    @router.get("/files/{file_path:path}")
    async def download_editable_file(file_path: str):
        try:
            path = await run_in_threadpool(editable_file_task_service.public_file_path, file_path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"error": "file not found"}) from exc
        return FileResponse(path, filename=path.name)

    @router.post("/v1/ppt/generations")
    async def create_ppt_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/ppt/generations", "gpt-5-5-thinking", "PPT生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_ppt,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    @router.post("/v1/psd/generations")
    async def create_psd_task(body: EditableFileTaskRequest, request: Request, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/v1/psd/generations", "gpt-5-5-thinking", "PSD生成任务", request_text=body.prompt), body.prompt)
        return await run_in_threadpool(
            editable_file_task_service.submit_psd,
            identity,
            client_task_id=body.client_task_id or "",
            prompt=body.prompt,
            base64_images=body.base64_images,
            base_url=resolve_image_base_url(request),
        )

    return router

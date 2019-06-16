import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any
import io
import aiohttp.multipart
import aiohttp.web

from PIL import Image
from nima.worker import WorkersConfig, init_workers, predict

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self, executor: ThreadPoolExecutor):
        self._loop = asyncio.get_event_loop()
        self._executor = executor

    def register(self, app: aiohttp.web.Application) -> None:
        app.add_routes(
            (aiohttp.web.post("/predict", self.handle_predict, name="predict"),)
        )
        app.add_routes((aiohttp.web.get("/ping", self.handle_ping),))

    @staticmethod
    async def handle_ping(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response()


    async def _save_image(self, field: aiohttp.multipart.BodyPartReader) -> Path:
        filename = field.filename
        size = 0
        file_path = self._path_to_save_images / f"{uuid.uuid4()}_{filename}"
        with open(file_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()  # 8192 bytes by default.
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)
        return file_path

    async def parse_and_save_image(
            self, reader: aiohttp.multipart.MultipartReader
    ) -> Tuple[Path, str]:
        start_time = time.monotonic()
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "box_type":
                box_type = await field.text()
            if field.name == "image":
                file_path = await self._save_image(field)

        end_time = time.monotonic()
        logger.info(
            f"save_image file_path = {file_path}, "
            f"box_type = {box_type}, "
            f"time = {end_time - start_time}"
        )

        image_path_with_box = Path(
            str(file_path).replace(file_path.suffix, f"_{box_type}_{file_path.suffix}")
        )

        file_path.rename(image_path_with_box)

        return image_path_with_box, box_type

    async def handle_predict(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        start = time.monotonic()
        form = await request.post()
        raw_data = form['file'].file.read()
        image = Image.open(io.BytesIO(raw_data))
        #
        executor = self._executor
        result = await self._loop.run_in_executor(
            executor, predict, image
        )
        end = time.monotonic()
        total_time = end - start
        result["total_time"] = total_time
        logger.info(f"total request time is {total_time}")
        return aiohttp.web.json_response(result)


@aiohttp.web.middleware
async def handle_exceptions(request: aiohttp.web.Request, handler) -> aiohttp.web.Response:  # type: ignore
    try:
        return await handler(request)
    except ValueError as e:
        payload = {"error": str(e)}
        return aiohttp.web.json_response(
            payload, status=aiohttp.web.HTTPBadRequest.status_code
        )
    except aiohttp.web.HTTPException:
        raise
    except Exception as e:
        msg_str = (
            f"Unexpected exception: {str(e)}. " f"Path with query: {request.path_qs}."
        )
        logging.exception(msg_str)
        payload = {"error": msg_str}
        return aiohttp.web.json_response(
            payload, status=aiohttp.web.HTTPInternalServerError.status_code
        )


async def create_models_app(executor: ThreadPoolExecutor,
                            models_app: aiohttp.web.Application
                            ) -> aiohttp.web.Application:
    models_handler = ModelHandler(executor)
    models_handler.register(models_app)
    return models_app


async def create_app(config: "Config") -> aiohttp.web.Application:
    app = aiohttp.web.Application(middlewares=[handle_exceptions])
    app["config"] = config

    executor = await init_workers(app, config.worker)
    app["executor"] = executor
    app = await create_models_app(executor=executor, models_app=app)
    return app


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    worker: WorkersConfig


def run_api(path_to_model_state: Path, port: int = 8080, host: str = '0.0.0.0') -> None:
    config = Config(server=ServerConfig(port=port, host=host), worker=WorkersConfig(path_to_model_state=path_to_model_state))
    logging.info("Loaded config: %r", config)
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app(config))
    aiohttp.web.run_app(app, host=config.server.host, port=config.server.port)

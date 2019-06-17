import asyncio
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
        app.add_routes((aiohttp.web.post("/predict", self.handle_predict, name="predict"),))
        app.add_routes((aiohttp.web.get("/ping", self.handle_ping),))

    @staticmethod
    async def handle_ping(request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.Response()

    async def handle_predict(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        start = time.monotonic()
        form = await request.post()
        raw_data = form["file"].file.read()
        image = Image.open(io.BytesIO(raw_data))
        #
        executor = self._executor
        result = await self._loop.run_in_executor(executor, predict, image)
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
        return aiohttp.web.json_response(payload, status=aiohttp.web.HTTPBadRequest.status_code)
    except aiohttp.web.HTTPException:
        raise
    except Exception as e:
        msg_str = f"Unexpected exception: {str(e)}. " f"Path with query: {request.path_qs}."
        logging.exception(msg_str)
        payload = {"error": msg_str}
        return aiohttp.web.json_response(payload, status=aiohttp.web.HTTPInternalServerError.status_code)


async def create_models_app(
    executor: ThreadPoolExecutor, models_app: aiohttp.web.Application
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


def run_api(path_to_model_state: Path, port: int = 8080, host: str = "0.0.0.0") -> None:
    config = Config(
        server=ServerConfig(port=port, host=host), worker=WorkersConfig(path_to_model_state=path_to_model_state)
    )
    logging.info("Loaded config: %r", config)
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app(config))
    aiohttp.web.run_app(app, host=config.server.host, port=config.server.port)

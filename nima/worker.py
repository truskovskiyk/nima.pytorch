import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from aiohttp import web

from nima.inference_model import InferenceModel

logger = logging.getLogger(__name__)
_model = None


def warm(path_to_model_state: Path) -> None:
    logger.info(f"warm model")
    global _model
    if _model is None:
        # load model
        _model = InferenceModel(path_to_model_state=path_to_model_state)
        logger.info(f"created model {_model}")


def clean() -> None:
    global _model
    _model = None


def predict(image: Image.Image,
            model: Optional[InferenceModel] = None
            ):
    if model is None:
        model = _model

    if model is None:
        raise RuntimeError("Model should be loaded first")
    result = model.predict_from_pil_image(image)
    return result


@dataclass(frozen=True)
class WorkersConfig:
    path_to_model_state: Path
    max_workers: int = 1



async def init_workers(app: web.Application, conf: WorkersConfig) -> ThreadPoolExecutor:
    n = conf.max_workers
    executor = ThreadPoolExecutor(max_workers=n)

    loop = asyncio.get_event_loop()
    run = loop.run_in_executor
    fs = [run(executor, warm, conf.path_to_model_state) for _ in range(0, n)]
    await asyncio.gather(*fs)

    async def close_executor(app: web.Application) -> None:
        fs = [run(executor, clean) for _ in range(0, n)]
        await asyncio.shield(asyncio.gather(*fs))
        executor.shutdown(wait=True)

    app.on_cleanup.append(close_executor)
    app["executor"] = executor
    return executor

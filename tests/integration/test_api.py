import io
from typing import NamedTuple, AsyncIterator
from pathlib import Path
import aiohttp
import pytest
from aiohttp import FormData
from aiohttp.web import HTTPOk
from PIL import Image
from nima.api import Config, ServerConfig, WorkersConfig, create_app


@pytest.fixture
def config(state_dict_path) -> Config:
    server_config = ServerConfig()
    workers_config = WorkersConfig(path_to_model_state=state_dict_path)
    return Config(server=server_config, worker=workers_config)


class ApiConfig(NamedTuple):
    host: str
    port: int

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_base_url(self) -> str:
        return self.endpoint

    @property
    def ping_url(self) -> str:
        return self.endpoint + "/ping"


@pytest.fixture
async def api(config: Config) -> AsyncIterator[ApiConfig]:
    app = await create_app(config)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    api_config = ApiConfig(host="0.0.0.0", port=8080)
    site = aiohttp.web.TCPSite(runner, api_config.host, api_config.port)
    await site.start()
    yield api_config
    await runner.cleanup()


@pytest.fixture
async def client() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


class TestModelApi:

    @pytest.mark.asyncio
    async def test_predict(self,
                                api: ApiConfig,
                                client: aiohttp.ClientSession,
                                image_file_obj: io.BytesIO
                                ) -> None:
        predict_url = api.model_base_url + "/predict"

        data = FormData()
        data.add_field(
            "file",
            image_file_obj,
            filename="test_image.jpg",
            content_type="image/img",
        )

        async with client.post(predict_url, data=data) as response:
            assert response.status == HTTPOk.status_code
            res_data = await response.json()
            assert 'mean_score' in res_data
            assert 'std_score' in res_data
            assert 'scores' in res_data
            assert 'total_time' in res_data



class TestApi:
    @pytest.mark.asyncio
    async def test_ping(self, api: ApiConfig, client: aiohttp.ClientSession) -> None:
        async with client.get(api.ping_url) as response:
            assert response.status == HTTPOk.status_code

import traceback

import click
import pytest
from click.testing import CliRunner

from nima.cli import prepare_dataset


def test_hello_world():
    @click.command()
    @click.argument("name")
    def hello(name):
        click.echo("Hello %s!" % name)

    runner = CliRunner()
    result = runner.invoke(hello, ["Peter"])
    print(result)
    assert result.exit_code == 0
    assert result.output == "Hello Peter!\n"


def test_prepare_dataset():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass


def test_get_image_score():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass


def test_train_model():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass


def test_validate_model():
    # runner = CliRunner()
    # result = runner.invoke(prepare_dataset, ['--path_to_ava_txt', 'test.csv'])
    # print(result.exception)
    # print(result.output)
    # print(result.exc_info[-1])
    # traceback.print_tb(result.exc_info[-1])
    pass

import pytest
import torch

from src.models import MODELS, CNNTransformer, ConvFrontEnd, DeepLOB, count_params


def test_front_end_shape():
    assert ConvFrontEnd()(torch.randn(2, 1, 100, 40)).shape == (2, 100, 96)


@pytest.mark.parametrize("name", MODELS)
def test_output_shape(name):
    model = MODELS[name]().eval()
    assert model(torch.randn(8, 1, 100, 40)).shape == (8, 3)


def test_parameter_budgets_match():
    n_lstm, n_transformer = count_params(DeepLOB()), count_params(CNNTransformer())
    assert abs(n_transformer - n_lstm) / n_lstm < 0.05

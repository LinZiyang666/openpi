import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer

# Both tokenizers download remote assets on construction
# (``download.maybe_download("gs://big_vision/paligemma_tokenizer.model")`` +
# ``AutoProcessor.from_pretrained``), so these tests need network / GCS access.
pytestmark = pytest.mark.env_dependent(
    reason="constructs tokenizers that download GCS / HuggingFace assets"
)


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)

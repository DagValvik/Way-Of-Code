import math

import pytest

import main as module


@pytest.fixture(scope="module")
def split_data():
    train_df, test_df = module.load_data("data/combined_sentiment_data.csv")
    return train_df, test_df


def test_load_data(split_data):
    """Tests if data loading and splitting works correctly."""
    train_df, test_df = split_data

    # Check total size and split ratio
    total_size = len(train_df) + len(test_df)
    assert total_size == 3309
    assert len(train_df) == math.floor(0.8 * 3309)  # ~80% of 3309
    assert len(test_df) == math.ceil(0.2 * 3309)  # ~20% of 3309

    # Check that we have the expected columns
    assert "sentence" in train_df.columns
    assert "sentiment" in train_df.columns

    # Check label distribution
    assert all(label in [0, 1] for label in train_df["sentiment"])
    assert all(label in [0, 1] for label in test_df["sentiment"])


def test_metrics():
    """Tests if evaluation metrics are calculated correctly."""
    # Test with sample predictions
    actual_labels = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    predictions = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]

    metrics = module.evaluate(predictions, actual_labels)

    # Check that we get the expected metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # Check specific values
    assert metrics["accuracy"] == pytest.approx(0.75, 1e-4)
    assert metrics["precision"] == pytest.approx(0.8333, 1e-4)
    assert metrics["recall"] == pytest.approx(0.7143, 1e-4)
    assert metrics["f1"] == pytest.approx(0.7692, 1e-4)

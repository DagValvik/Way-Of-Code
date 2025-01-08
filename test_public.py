import pytest

import main as module


@pytest.fixture(scope="module")
def split_data():
    return module.load_data("data/combined_sentiment_data.csv")


@pytest.fixture(scope="module")
def processed_train_data(split_data):
    train_contents, _, _, _ = split_data
    return module.preprocess_multiple(train_contents)


@pytest.fixture(scope="module")
def processed_test_data(split_data):
    _, _, test_contents, _ = split_data
    return module.preprocess_multiple(test_contents)


@pytest.fixture(scope="module")
def feature_vectors(processed_train_data, processed_test_data):
    return module.extract_features(processed_train_data, processed_test_data)


@pytest.fixture(scope="module")
def classifier(split_data, feature_vectors):
    train_contents, train_labels, _, _ = split_data
    train_vectors, _ = feature_vectors
    return module.train(train_vectors, train_labels)


def test_load_data(split_data):
    """Tests if data loading and splitting works correctly."""
    train_contents, train_labels, test_contents, test_labels = split_data

    # Check total size and split ratio
    total_size = len(train_contents) + len(test_contents)
    assert total_size == 3309
    assert len(train_contents) == 2647  # ~80% of 3309
    assert len(test_contents) == 662  # ~20% of 3309

    # Check first example
    assert (
        train_contents[0]
        == "Our waiter was very attentive,  friendly,  and informative."
    )
    assert train_labels[0] == 1

    # Check label distribution
    assert all(label in [0, 1] for label in train_labels)
    assert all(label in [0, 1] for label in test_labels)


def test_preprocess(split_data):
    """Tests if preprocessing works correctly."""
    train_contents, _, _, _ = split_data
    preprocessed_example = module.preprocess(train_contents[1]).split()
    assert "situations" in preprocessed_example
    assert "," not in preprocessed_example
    assert "1.)" not in preprocessed_example


def test_preprocess_multiple(processed_train_data, processed_test_data):
    """Tests if multiple document preprocessing works."""
    assert len(processed_train_data) + len(processed_test_data) == 3309


def test_extract_features(feature_vectors):
    """Tests if feature extraction produces valid feature vectors."""
    train_vectors, test_vectors = feature_vectors
    assert train_vectors.shape[0] + test_vectors.shape[0] == 3309
    assert train_vectors.shape[1] > 0
    assert test_vectors.shape[1] > 0


def test_train(classifier, feature_vectors):
    """Tests if model training produces valid predictions."""
    _, test_vectors = feature_vectors
    predictions = classifier.predict(test_vectors)
    assert len(predictions) == test_vectors.shape[0]


def test_validation_evaluation():
    """Tests if evaluation metrics are calculated correctly."""
    test_labels = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    ground_truth = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    recall, precision, f1, accuracy = module.evaluate(test_labels, ground_truth)
    assert recall == pytest.approx(0.8333, 1e-4)
    assert precision == pytest.approx(0.7143, 1e-4)
    assert f1 == pytest.approx(0.7692, 1e-4)
    assert accuracy == 0.75

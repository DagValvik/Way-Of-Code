import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(path: str):
    """Loads and splits data"""
    df = pd.read_csv(path)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df


def train(model, train_loader, device):
    """Trains the model following the article's approach"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(2):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    return model


def predict(model, test_loader, device):
    """Makes predictions using the trained model"""
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    return predictions, actual_labels


def evaluate(predictions, actual_labels):
    """Returns a dictionary of evaluation metrics"""
    recall, precision, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average="binary"
    )
    accuracy = accuracy_score(actual_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Load and prepare data
    print("Loading data...")
    train_df, test_df = load_data("data/combined_sentiment_data.csv")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Create datasets
    train_dataset = SentimentDataset(
        texts=train_df["sentence"].values,
        labels=train_df["sentiment"].values,
        tokenizer=tokenizer,
    )
    test_dataset = SentimentDataset(
        texts=test_df["sentence"].values,
        labels=test_df["sentiment"].values,
        tokenizer=tokenizer,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    # Train
    print("Training...")
    model = train(model, train_loader, device)

    # Predict and evaluate
    print("Evaluating...")
    predictions, actual_labels = predict(model, test_loader, device)

    # Calculate metrics
    metrics = evaluate(predictions, actual_labels)

    print(f"Recall:\t{metrics['recall']:.4f}")
    print(f"Precision:\t{metrics['precision']:.4f}")
    print(f"F1:\t{metrics['f1']:.4f}")
    print(f"Accuracy:\t{metrics['accuracy']:.4f}")

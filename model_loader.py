# model_loader.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def save_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")
    print("✅ Model with 3 sentiment classes saved!")

if __name__ == "__main__":
    save_model()
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity
)
import ast
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

os.environ.pop("OPENAI_API_KEY", None)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
)
# ----------------------------
# 1. Load RAG Output Dataset
# ----------------------------
rag_df = pd.read_csv("rag_output_dataset.csv", sep=";")

# ----------------------------
# 2. Define Ground Truth (from your pasted data)
# ----------------------------
gt_df = pd.read_csv("questions_dataset.csv", sep=";")

# ----------------------------
# 3. Helper: Parse JSON flashcards → plain text
# ----------------------------
def json_to_text(json_str):
    if not isinstance(json_str, str) or not json_str.strip():
        return ""
    try:
        # Fix double-escaped quotes if needed
        json_str = json_str.replace('""', '"')
        items = json.loads(json_str)
        if isinstance(items, list):
            return "\n".join([f"{item['term']}: {item['definition']}" for item in items])
        else:
            return str(items)
    except Exception as e:
        print(f"Parse error: {e}")
        return ""

# ----------------------------
# 4. Merge & Preprocess
# ----------------------------
df = pd.merge(rag_df, gt_df, on="question", how="inner")

df["answer"] = df["rag_output"].apply(json_to_text)
df["reference"] = df["ground_truth_output"].apply(json_to_text)

# Parse contexts (string representation of list → actual list)
def parse_contexts(ctx_str):
    try:
        # Clean and parse
        ctx_str = ctx_str.strip().replace('""', '"')
        return ast.literal_eval(ctx_str)
    except:
        return [""]

df["contexts"] = df["contexts"].apply(parse_contexts)

# ----------------------------
# 5. Create Dataset for RAGAS
# ----------------------------
dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "reference"]])

# ----------------------------
# 6. Run Evaluation
# ----------------------------
metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity
]

print("Running RAGAS evaluation...")
results = evaluate(
    dataset, 
    metrics=metrics, 
    llm=llm,
    embeddings=embeddings
    )
print("\n✅ Evaluation Results:")
print(results)
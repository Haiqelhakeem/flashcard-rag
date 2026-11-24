import json
import ast
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Load your data
df = pd.read_csv("rag_output_dataset.csv", sep=";", quotechar='"')

# Convert rag_output JSON to readable answer
def json_to_answer(json_str):
    try:
        items = json.loads(json_str)
        return " ".join([f"{item['term']}: {item['definition']}" for item in items])
    except:
        return ""

# Convert contexts string to list
def str_to_list(list_str):
    try:
        return ast.literal_eval(list_str)
    except:
        return []

df["answer"] = df["rag_output"].apply(json_to_answer)
df["contexts"] = df["contexts"].apply(str_to_list)

# Create dataset
dataset = Dataset.from_pandas(df[["question", "answer", "contexts"]])

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

# View scores
print(results)
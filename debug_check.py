from datasets import load_dataset
import json
from unsloth import UnslothTrainer, UnslothTrainingArguments, FastLanguageModel

def main():
    squad  = load_dataset("json", data_files="cosmetology_dataset.jsonl", split="train")

    
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=1024,  # сокращаем контекст
        load_in_4bit=True,
        dtype=None
    )

# 2. Функция подготовки, которая объединяет контекст + нужный ответ
    def preprocess_fn(examples):
        inputs = []
        for messages, response in zip(examples["messages"], examples["response"]):
            # Собираем контекст
            context = ""
            for m in messages:
                context += f"{m['role']}: {m['content']}\n"
            # Склеиваем prompt + ответ (response)
            response_str = json.dumps(response, ensure_ascii=False)
            text = context + "Assistant: " + response_str
            inputs.append(text)
        tokenized = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    squad_processed = squad.map(preprocess_fn, batched=True, remove_columns=["messages","response"])
    squad_processed.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    for i in range(5):
        item = squad_processed[i]
        print("=== Example", i, "===")
        print("input_ids:", item["input_ids"][:20], "...")       # первые 20 токенов, для наглядности
        print("attention_mask:", item["attention_mask"][:20], "...")
        print("labels:", item["labels"][:20], "...")
        print()

    # Если у tokenizer есть метод decode:
    for i in range(5):
        input_ids = squad_processed[i]["input_ids"]
        # уберём padding / токенизированные до max_len
        # найдём где input_ids == pad_token_id и отрежем
        # но для простоты:
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print("Decoded:", text)
        print()

if __name__ == "__main__":
    main()
# finetune_llama3_8b_minimal.py

import torch
import json 
from datasets import load_dataset
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments, FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from unsloth import is_bfloat16_supported


def main():
    # 0. (опционально) очистка кэша до начала
    torch.cuda.empty_cache()

    # 1. Загрузка датасета
    squad  = load_dataset("json", data_files="cosmetology_dataset.jsonl", split="train")

    


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

    


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=1024,  # сокращаем контекст
        load_in_4bit=True,
        dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout= 0.01,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    squad_processed = squad.map(preprocess_fn, batched=True, remove_columns=["messages","response"])
    squad_processed.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)




    trainer = TrainingArguments(
        model=model,
        tokenizer=tokenizer,
        train_dataset=squad_processed,
        dataset_text_field="input_text",
        max_seq_length=1024,
        data_collator=data_collator,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            num_train_epochs=3,
            learning_rate=1e-5,
            embedding_learning_rate=1e-5,
            fp16 = not is_bfloat16_supported(), # Use 16-bit floating point if bfloat16 isn't supported.
            bf16 = is_bfloat16_supported(),     #use bf16 if hardware support
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear", #Cosine scheduler to adjust the learning rate over time.
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    # Train
    trainer.train()
    trainer.save_model("./finetuned_llama3_cosmetology_model")

if __name__ == "__main__":
    main()

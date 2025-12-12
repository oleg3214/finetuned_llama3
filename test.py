import torch
import sys
from unsloth import FastLanguageModel
from colorama import init, Fore, Style
from peft import PeftModel

# Инициализация цветного вывода
init(autoreset=True)

def load_model():
    """Загрузка модели с LoRA-адаптерами"""
    print(Fore.CYAN + "🚀 Загружаю модель...")
    
    try:
        # 1. Загружаем базовую модель
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            max_seq_length=1024,
            load_in_4bit=True,
            dtype=None,
        )
        
        # 2. Загружаем обученные адаптеры
        print("Загружаю LoRA-адаптеры через PeftModel...")
        model = PeftModel.from_pretrained(
            base_model,
            "./finetuned_llama3_cosmetology_model",  # ← Ваша папка с адаптерами
            adapter_name="cosmetology_lora"
        )
        
        # 3. Переводим в режим инференса
        model.eval()
        
        print(Fore.GREEN + "✅ Модель загружена успешно!")
        return model, tokenizer
        
    except Exception as e:
        print(Fore.RED + f"❌ Ошибка загрузки модели: {e}")
        print(Fore.YELLOW + "Проверьте путь: ./finetuned_llama3_cosmetology_model")
        return None, None

def chat_loop(model, tokenizer):
    """Основной цикл чата"""
    print(Fore.CYAN + "\n" + "="*60)
    print(Fore.CYAN + "💬 ЧАТ С МОДЕЛЬЮ (Ctrl+C для выхода)")
    print(Fore.CYAN + "="*60)
    
    # Системный промпт
    system_prompt = (
        "Ты — опытный косметолог-ассистент. У тебя есть инструменты: "
        "\"calculator\" (для расчётов) и \"call_DB\" (для работы с базой данных). "
        "Отвечай подробно и профессионально на русском языке.\n\n"
    )
    
    # История диалога (можно ограничить для экономии памяти)
    conversation = []
    
    while True:
        try:
            # Получаем вопрос от пользователя
            print(Fore.YELLOW + "\n[Введите вопрос или 'выход'/quit]:")
            user_input = input(Fore.WHITE + "👤 Вы: ").strip()
            
            # Проверка на выход
            if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                print(Fore.CYAN + "\n👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            # Добавляем в историю
            conversation.append(f"user: {user_input}")
            
            # Формируем полный промпт
            full_prompt = system_prompt + "\n".join(conversation[-4:]) + "\nassistant: "
            
            # Токенизация
            inputs = tokenizer(
                [full_prompt],
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to("cuda")
            
            # Генерация ответа с индикатором
            print(Fore.BLUE + "\n🤖 Модель генерирует ответ...", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,           # Максимальная длина ответа
                    temperature=0.7,              # Креативность
                    do_sample=True,               # Сэмплирование
                    top_p=0.9,                    # Качество текста
                    repetition_penalty=1.1,       # Штраф за повторы
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            print(Fore.GREEN + " ✅")
            
            # Декодируем ответ
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ ассистента (последний)
            assistant_response = full_response.split("assistant:")[-1].strip()
            
            # Выводим ответ красиво
            print(Fore.GREEN + f"🤖 Ассистент: {assistant_response}")
            
            # Добавляем ответ в историю
            conversation.append(f"assistant: {assistant_response}")
            
            # Ограничиваем историю (последние 3 пары)
            if len(conversation) > 6:
                conversation = conversation[-6:]
                
        except KeyboardInterrupt:
            print(Fore.CYAN + "\n\n👋 Диалог завершен по запросу пользователя.")
            break
        except Exception as e:
            print(Fore.RED + f"\n⚠️ Ошибка: {e}")
            continue

def quick_test(model, tokenizer):
    """Быстрый тест модели перед чатом"""
    print(Fore.CYAN + "\n🧪 Быстрая проверка модели...")
    
    test_prompts = [
        "Привет! Как дела?",
        "Что такое кислотный пилинг?",
        "Сколько будет 15 + 27?",
    ]
    
    for prompt in test_prompts:
        full_prompt = f"user: {prompt}\nassistant: "
        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("assistant:")[-1].strip()
        
        print(Fore.WHITE + f"\nQ: {prompt}")
        print(Fore.GREEN + f"A: {answer[:100]}..." if len(answer) > 100 else f"A: {answer}")
    
    print(Fore.GREEN + "\n✅ Модель работает корректно!")

def main():
    """Основная функция"""
    print(Fore.CYAN + "="*60)
    print(Fore.CYAN + "🤖 ИНТЕРАКТИВНЫЙ ЧАТ С ОБУЧЕННОЙ МОДЕЛЬЮ")
    print(Fore.CYAN + "="*60)
    
    # Загружаем модель
    model, tokenizer = load_model()
    if model is None:
        return
    
    # Быстрая проверка
    quick_test(model, tokenizer)
    
    # Запускаем чат
    chat_loop(model, tokenizer)
    
    # Очистка памяти при выходе
    torch.cuda.empty_cache()
    print(Fore.CYAN + "\n💾 Память очищена.")

if __name__ == "__main__":
    # Устанавливаем поддержку UTF-8 для Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()
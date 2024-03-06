from ctransformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", model_file="openhermes-2.5-mistral-7b.Q5_K_M.gguf", model_type="mistral", gpu_layers=3000, hf=True)
tokenizer = AutoTokenizer.from_pretrained(model)


# Definir função de predição
def predict(input_text):
    # Tokenizar texto de entrada
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Realizar previsão
    outputs = model(**inputs).logits
    predicted_label_id = torch.argmax(outputs, dim=1).item()

    return predicted_label_id


# Criar chatbot
def chatbot():
    print(
        "Olá! Eu sou um assistente inteligente especializado em responder perguntas sobre os livros da editora Shalom. Digite sua pergunta ou 'fim' para encerrar a conversa.")

    while True:
        user_input = input("Você: ")

        if user_input.lower() == "fim":
            break

        prediction = predict(user_input)
        answer = tokenizer.decode(inputs['input_ids'][:prediction], skip_special_tokens=True)
        print("Assistente:", answer)

chatbot()
# print(llm("Give me a complete example of code that uses openhemes 2.5 to answer different questions that a user may ask in Brazilian Portuguese"))
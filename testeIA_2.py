# from ctransformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch_rand1 = torch.rand(100, 100, 100, 100).to(device)
torch_rand2 = torch.rand(100, 100, 100, 100).to(device)
np_rand1 = torch.rand(100, 100, 100, 100)
np_rand2 = torch.rand(100, 100, 100, 100)

start_time = time.time()
rand = (torch_rand1 @ torch_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f'{elapsed_time:.8f}')

start_time = time.time()
rand2 = np.multiply(np_rand1, np_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f'{elapsed_time:.8f}')




# model = AutoModelForCausalLM.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GGUF", model_file="openhermes-2.5-mistral-7b.Q5_K_M.gguf", model_type="mistral", gpu_layers=1500, hf=True)
# tokenizer = AutoTokenizer.from_pretrained(model)


# Definir função de predição
# def predict(input_text):
    # Tokenizar texto de entrada
#    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#
    # Realizar previsão
#    outputs = model(**inputs).logits
#    predicted_label_id = torch.argmax(outputs, dim=1).item()
#
#    return predicted_label_id
#
#
# Criar chatbot
# def chatbot():
#    print(
#        "Olá! Eu sou um assistente inteligente especializado em responder perguntas sobre os livros da editora Shalom. Digite sua pergunta ou 'fim' para encerrar a conversa.")
#
#    while True:
#        user_input = input("Você: ")
#
#        if user_input.lower() == "fim":
#            break
#
#        prediction = predict(user_input)
#        answer = tokenizer.decode(inputs['input_ids'][:prediction], skip_special_tokens=True)
#        print("Assistente:", answer)
#
# chatbot()
# print(llm("Give me a complete example of code that uses openhemes 2.5 to answer different questions that a user may ask in Brazilian Portuguese"))
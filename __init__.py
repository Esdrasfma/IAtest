
from ctransformers import AutoTokenizer, AutoModelForCausalLM
import textwrap
import torch

torch.set_default_device('cuda')

model_name = 'TheBloke/OpenHermes-2.5-Mistral-7B-GGUF'

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto").eval()

tokenizer = AutoTokenizer.from_pretrained(model)


def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def generate(input_text, system_prompt="", max_length=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                              return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), max_length=max_length)

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    wrapped_text = wrap_text(response)
    print(wrapped_text)



generate('who are you?', max_length=256)


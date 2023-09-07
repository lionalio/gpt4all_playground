import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
model_score = "microsoft/DialogRPT-updown"   
tokenizer_rpt = AutoTokenizer.from_pretrained(model_score)
model_rpt = AutoModelForSequenceClassification.from_pretrained(model_score)


def get_candidates(input_):
    # Embedding using tokenize
    input_1 = tokenizer.encode(input_, tokenizer.eos_token, return_tensors='pt')

    # Modeling
    outputs = model.generate(
        input_1,
        max_length=200,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True
    )

    # Decode the output using tokenizer
    ls_output = []
    for i in range(5):
        m = format(tokenizer.decode(outputs[:, input_1.shape[-1]:][i], skip_special_tokens=True))
        ls_output.append(m)

    return ls_output, outputs


def get_response(context, candidates):
    res = model_rpt(context, return_dict=True)
    score = torch.sigmoid(res.logits)

    ret = candidates[torch.argmax(score)]

    return ret


if __name__ == '__main__':
    while 1:
        input_ = input('user > ')
        if 'bye' in input_.lower():
            break

        candidates, output = get_candidates(input_)
        response = get_response(output, candidates)
        print('Bot >', response)
---
license: llama2
language:
- it
tags:
- text-generation-inference
---
<img src="https://i.ibb.co/6mHSRm3/llamantino53.jpg" alt="llamantino53" border="0" width="200px">

# Model Card for LLaMAntino-2-chat-7b-UltraChat-ITA
*Last Update: 08/01/2024*<br>*Example of Use*: [Colab Notebook](https://colab.research.google.com/drive/1lCQ7MqSNKILsIncNYhdN_yqzSvl4akat?usp=sharing)

<hr>


## Model description

<!-- Provide a quick summary of what the model is/does. -->

**LLaMAntino-2-chat-7b-UltraChat** is a *Large Language Model (LLM)* that is an instruction-tuned version of **LLaMAntino-2-chat-7b** (an italian-adapted **LLaMA 2 chat**). 
This model aims to provide Italian NLP researchers with an improved model for italian dialogue use cases.

The model was trained using *QLora* and using as training data [UltraChat](https://github.com/thunlp/ultrachat) translated to the italian language using [Argos Translate](https://pypi.org/project/argostranslate/1.4.0/). 
If you are interested in more details regarding the training procedure, you can find the code we used at the following link:
- **Repository:** https://github.com/swapUniba/LLaMAntino

**NOTICE**: the code has not been released yet, we apologize for the delay, it will be available asap!

- **Developed by:** Pierpaolo Basile, Elio Musacchio, Marco Polignano, Lucia Siciliani, Giuseppe Fiameni, Giovanni Semeraro
- **Funded by:** PNRR project FAIR - Future AI Research
- **Compute infrastructure:** [Leonardo](https://www.hpc.cineca.it/systems/hardware/leonardo/) supercomputer
- **Model type:** LLaMA-2-chat
- **Language(s) (NLP):** Italian
- **License:** Llama 2 Community License 
- **Finetuned from model:** [swap-uniba/LLaMAntino-2-chat-7b-hf-ITA](https://huggingface.co/swap-uniba/LLaMAntino-2-chat-7b-hf-ITA)

## Prompt Format

This prompt format based on the [LLaMA 2 prompt template](https://gpus.llm-utils.org/llama-2-prompt-template/) adapted to the italian language was used:

```python
" [INST]<<SYS>>\n" \
"Sei un assistente disponibile, rispettoso e onesto. " \
"Rispondi sempre nel modo piu' utile possibile, pur essendo sicuro. " \
"Le risposte non devono includere contenuti dannosi, non etici, razzisti, sessisti, tossici, pericolosi o illegali. " \
"Assicurati che le tue risposte siano socialmente imparziali e positive. " \
"Se una domanda non ha senso o non e' coerente con i fatti, spiegane il motivo invece di rispondere in modo non corretto. " \
"Se non conosci la risposta a una domanda, non condividere informazioni false.\n" \
"<</SYS>>\n\n" \
f"{user_msg_1}[/INST] {model_answer_1} </s> <s> [INST]{user_msg_2}[/INST] {model_answer_2} </s> ... <s> [INST]{user_msg_N}[/INST] {model_answer_N} </s>"
```

We recommend using the same prompt in inference to obtain the best results!

## How to Get Started with the Model

Below you can find an example of model usage:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

user_msg = "Ciao! Come stai?"

prompt = " [INST]<<SYS>>\n" \
         "Sei un assistente disponibile, rispettoso e onesto. " \
         "Rispondi sempre nel modo piu' utile possibile, pur essendo sicuro. " \
         "Le risposte non devono includere contenuti dannosi, non etici, razzisti, sessisti, tossici, pericolosi o illegali. " \
         "Assicurati che le tue risposte siano socialmente imparziali e positive. " \
         "Se una domanda non ha senso o non e' coerente con i fatti, spiegane il motivo invece di rispondere in modo non corretto. " \
         "Se non conosci la risposta a una domanda, non condividere informazioni false.\n" \
         "<</SYS>>\n\n" \
         f"{user_msg}[/INST]"

pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # langchain expects the full text
    task='text-generation',
    max_new_tokens=512, # max number of tokens to generate in the output
    temperature=0.8  #temperature for more or less creative answers
)

# Method 1
sequences = pipe(text)
for seq in sequences:
    print(f"{seq['generated_text']}")

# Method 2
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids=input_ids, max_length=512)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy()[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
```

If you are facing issues when loading the model, you can try to load it **Quantized**:

```python
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
```

*Note*:
1) The model loading strategy above requires the [*bitsandbytes*](https://pypi.org/project/bitsandbytes/) and [*accelerate*](https://pypi.org/project/accelerate/) libraries
2) The Tokenizer, by default, adds at the beginning of the prompt the '\<BOS\>' token. If that is not the case, add as a starting token the *\<s\>* string.
   
## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

*Coming soon*!

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

If you use this model in your research, please cite the following:

```bibtex
@misc{basile2023llamantino,
      title={LLaMAntino: LLaMA 2 Models for Effective Text Generation in Italian Language}, 
      author={Pierpaolo Basile and Elio Musacchio and Marco Polignano and Lucia Siciliani and Giuseppe Fiameni and Giovanni Semeraro},
      year={2023},
      eprint={2312.09993},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

*Notice:* Llama 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. [*License*](https://ai.meta.com/llama/license/)

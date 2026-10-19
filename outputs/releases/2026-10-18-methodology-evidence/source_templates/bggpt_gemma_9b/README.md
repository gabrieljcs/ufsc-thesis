---
library_name: transformers
tags:
- gemma2
- instruct
- bggpt
- insait
license: gemma
language:
- bg
- en
base_model:
- google/gemma-2-9b-it
- google/gemma-2-9b
pipeline_tag: text-generation
---

# INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0

![image/png](https://cdn-uploads.huggingface.co/production/uploads/637e1f8cf7e01589cc17bf7e/p6d0YFHjWCQ3S12jWqO1m.png)

INSAIT introduces **BgGPT-Gemma-2-9B-IT-v1.0**, a state-of-the-art Bulgarian language model based on **google/gemma-2-9b** and **google/gemma-2-9b-it**.
BgGPT-Gemma-2-9B-IT-v1.0 is **free to use** and distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
This model was created by [`INSAIT`](https://insait.ai/), part of Sofia University St. Kliment Ohridski, in Sofia, Bulgaria.

# Model description

The model was built on top of Google’s Gemma 2 9B open models.
It was continuously pre-trained on around 100 billion tokens (85 billion in Bulgarian) using the Branch-and-Merge strategy INSAIT presented at [EMNLP’24](https://aclanthology.org/2024.findings-emnlp.1000/), 
allowing the model to gain outstanding Bulgarian cultural and linguistic capabilities while retaining its English performance. 
During the pre-training stage, we use various datasets, including Bulgarian web crawl data, freely available datasets such as Wikipedia, a range of specialized Bulgarian datasets sourced by the INSAIT Institute,
and machine translations of popular English datasets.
The model was then instruction-fine-tuned on a newly constructed Bulgarian instruction dataset created using real-world conversations.
For more information check our [blogpost](https://models.bggpt.ai/blog/).

# Benchmarks and Results

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65fefdc282708115868203aa/5knpdR-QDSuM3WlpRxe-M.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/65fefdc282708115868203aa/TY8F34DpUf7uXbsFVywn2.png)

We evaluate our models on a set of standard English benchmarks, a translated version of them in Bulgarian, as well as, Bulgarian specific benchmarks we collected:

- **Winogrande challenge**: testing world knowledge and understanding
- **Hellaswag**: testing sentence completion
- **ARC Easy/Challenge**: testing logical reasoning
- **TriviaQA**: testing trivia knowledge
- **GSM-8k**: solving multiple-choice questions in high-school mathematics
- **Exams**: solving high school problems from natural and social sciences
- **MON**: contains exams across various subjects for grades 4 to 12


These benchmarks test logical reasoning, mathematics, knowledge, language understanding and other skills of the models and are provided at https://github.com/insait-institute/lm-evaluation-harness-bg.
The graphs above show the performance of BgGPT 9B and BgGPT 27B compared to other large open models. The results show the excellent abilities of both 9B and 27B models in Bulgarian, which allow them to **outperform much larger models**,
including Alibaba’s Qwen 2.5 72B and Meta’s Llama3.1 70B. Further, both BgGPT 9B and BgGPT 27B **significantly improve upon the previous version of BgGPT** based on Mistral-7B ([BgGPT-7B-Instruct-v0.2](https://huggingface.co/INSAIT-Institute/BgGPT-7B-Instruct-v0.2), shown in grey in the figure). 
Finally, our models retain the **excellent English performance** inherited from the original Google Gemma 2 models upon which they are based.


# Use in 🤗 Transformers
First install the latest version of the transformers library:
```
pip install -U 'transformers[torch]'
```
Then load the model in transformers:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
```

# Recommended Parameters

For optimal performance, we recommend the following parameters for text generation, as we have extensively tested our model with them:

```python
from transformers import GenerationConfig

generation_params = GenerationConfig(
    max_new_tokens=2048,              # Choose maximum generation tokens
    temperature=0.1,
    top_k=25,
    top_p=1,
    repetition_penalty=1.1,
    eos_token_id=[1,107],
    do_sample=True
)
```

In principle, increasing temperature should work adequately as well.

# Instruction format

In order to leverage instruction fine-tuning, your prompt should begin with a beginning-of-sequence token `<bos>` and be formatted in the Gemma 2 chat template. `<bos>` should only be the first token in a chat sequence.

E.g.
```
<bos><start_of_turn>user
Кога е основан Софийският университет?<end_of_turn>
<start_of_turn>model
 
```

This format is also available as a [chat template](https://huggingface.co/docs/transformers/main/chat_templating) via the `apply_chat_template()` method:

```python
tokenizer = AutoTokenizer.from_pretrained(
    "INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0",
    use_default_system_prompt=False,
)

messages = [
    {"role": "user", "content": "Кога е основан Софийският университет?"},
]

input_ids = tokenizer.apply_chat_template(
  messages,
  return_tensors="pt",
  add_generation_prompt=True,
  return_dict=True
)

outputs = model.generate(
  **input_ids,
  generation_config=generation_params
)
print(tokenizer.decode(outputs[0]))

```

**Important Note:** Models based on Gemma 2 such as BgGPT-Gemma-2-9B-IT-v1.0 do not support flash attention. Using it results in degraded performance.

# Use with vLLM

Example usage with vLLM:

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0",
    use_default_system_prompt=False,
)

sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=0.1,
    top_k=25,
    top_p=1,
    repetition_penalty=1.1,
    stop_token_ids=[1, 107],
)

llm = LLM(
    model="INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0",
    dtype="bfloat16",
    enforce_eager=True
)

messages = [
    {"role": "user", "content": "Кога е основан Софийският университет?"},
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

input_ids = tokenizer(
    formatted_prompt,
    add_special_tokens=False
).input_ids

prompt = TokensPrompt(prompt_token_ids=input_ids)

output = llm.generate(
    prompt,
    sampling_params
)

generated_text = output[0].outputs[0].text
print(generated_text)
```

# Use with GGML / llama.cpp

The model and instructions for usage in GGUF format are available at [INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0-GGUF](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0-GGUF).

# Community Feedback

We welcome feedback from the community to help improve BgGPT. If you have suggestions, encounter any issues, or have ideas for improvements, please:
- Share your experience using the model through Hugging Face's community discussion feature or
- Contact us at [bggpt@insait.ai](mailto:bggpt@insait.ai)

Your real-world usage and insights are valuable in helping us optimize the model's performance and behaviour for various use cases.

# Summary
- **Finetuned from:** [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it); [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b);
- **Model type:** Causal decoder-only transformer language model
- **Language:** Bulgarian and English
- **Contact:** [bggpt@insait.ai](mailto:bggpt@insait.ai)
- **License:** BgGPT is distributed under [Gemma Terms of Use](https://huggingface.co/INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0/raw/main/LICENSE)
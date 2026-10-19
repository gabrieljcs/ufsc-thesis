---
extra_gated_prompt: |
  To access the model you need to belong to an European university or research organization 
  AND have a valid email adress correlating to the university or research organization you belong to. 
  You agree to use the model for research purposes only.
  To read more visit
  [https://www.ai.se/en/project/gpt-sw3](https://www.ai.se/en/project/gpt-sw3).
extra_gated_description: |
  We process new request once a week. 
  No requests will be processed during week 28-33.
extra_gated_fields:
  Full name: text
  Organization/university email adress: text
  Company/organization: text
  Country: country
  I agree to use this model for research purposes ONLY: checkbox

license: other
datasets:
- laion/OIG
- databricks/databricks-dolly-15k
- OpenAssistant/oasst1
language:
- da
- sv
- 'no'
- en
- is
base_model: AI-Sweden-Models/gpt-sw3-6.7b-v2
---
# Model description
[AI Sweden](https://huggingface.co/AI-Sweden-Models/)   
**Base models**  
[GPT-Sw3 126M](https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/) | [GPT-Sw3 356M](https://huggingface.co/AI-Sweden-Models/gpt-sw3-356m/) | [GPT-Sw3 1.3B](https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b/)  
[GPT-Sw3 6.7B](https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b/) | [GPT-Sw3 6.7B v2](https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b-v2/) | [GPT-Sw3 20B](https://huggingface.co/AI-Sweden-Models/gpt-sw3-20b/)  
[GPT-Sw3 40B](https://huggingface.co/AI-Sweden-Models/gpt-sw3-40b/)  
**Instruct models**  
[GPT-Sw3 126M Instruct](https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m-instruct/) | [GPT-Sw3 356M Instruct](https://huggingface.co/AI-Sweden-Models/gpt-sw3-356m-instruct/) | [GPT-Sw3 1.3B Instruct](https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b-instruct/)  
[GPT-Sw3 6.7B v2 Instruct](https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct/) | [GPT-Sw3 20B Instruct](https://huggingface.co/AI-Sweden-Models/gpt-sw3-20b-instruct/)  
**Quantized models**  
[GPT-Sw3 6.7B v2 Instruct 4-bit gptq](https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct-4bit-gptq) | [GPT-Sw3 20B Instruct 4-bit gptq](https://huggingface.co/AI-Sweden-Models/gpt-sw3-20b-instruct-4bit-gptq)  

GPT-SW3 is a collection of large decoder-only pretrained transformer language models that were developed by AI Sweden in collaboration with RISE and the WASP WARA for Media and Language. GPT-SW3 has been trained on a dataset containing 320B tokens in Swedish, Norwegian, Danish, Icelandic, English, and programming code. The model was pretrained using a causal language modeling (CLM) objective utilizing the NeMo Megatron GPT implementation.

The `instruct` models were finetrained on instruction data using both chat and raw text formats.

# Intended use
GPT-SW3 is an autoregressive large language model that is capable of generating coherent text in 5 different languages, and 4 programming languages. GPT-SW3 can also be instructed to perform text tasks that it has not been explicitly trained for, by casting them as text generation tasks.

# Limitations
Like other large language models for which the diversity (or lack thereof) of training data induces downstream impact on the quality of our model, GPT-SW3 has limitations in terms of for example bias and safety. GPT-SW3 can also have quality issues in terms of generation diversity and hallucination. By releasing with the modified RAIL license, we also hope to increase communication, transparency, and the study of large language models. The model may: overrepresent some viewpoints and underrepresent others, contain stereotypes, generate hateful, abusive, violent, discriminatory or prejudicial language. The model may make errors, including producing incorrect information as if it were factual, it may generate irrelevant or repetitive outputs, and content that may not be appropriate for all settings, including sexual content.

# How to use
To be able to access the model from Python, since this is a private repository, you have to log in with your access token. This can be done with `huggingface-cli login`, see [HuggingFace Quick Start Guide](https://huggingface.co/docs/huggingface_hub/quick-start#login) for more information.

The following code snippet loads our tokenizer & model, and uses the GPU if available.

```python
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize Variables
model_name = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
prompt = "Träd är fina för att"

# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.to(device)
```

Generating text using the `generate` method is done as follows:
```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

generated_token_ids = model.generate(
    inputs=input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_p=1,
)[0]

generated_text = tokenizer.decode(generated_token_ids)    
```

The chat format used during data-preprocessing takes the form:
```
<|endoftext|><s>
User:
Jag tycker träd är fina
<s>
Bot:
Kul att du tycker det!
<s>
...
```

The procedure to generate text is the same as before:

```python
prompt = """
<|endoftext|><s>
User:
Varför är träd fina?
<s>
Bot:
""".strip()

input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

generated_token_ids = model.generate(
    inputs=input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_p=1,
)[0]

generated_text = tokenizer.decode(generated_token_ids)    
```

Generating text using the `generate` method is done as follows:
```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

generated_token_ids = model.generate(
    inputs=input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.6,
    top_p=1,
)[0]

A convenient alternative to the `generate` method is the HuggingFace pipeline, which handles most of the work for you:
```python
generator = pipeline('text-generation', tokenizer=tokenizer, model=model, device=device)
generated = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.6, top_p=1)[0]["generated_text"]
```

# Maintenance
- Who is supporting/hosting/maintaining the dataset? AI Sweden at Lindholmen Science Park AB.
- How can the owner/curator/manager of the dataset be contacted (e.g., email address)? nlu@ai.se
- Is there an erratum? If so, please provide a link or other access point. N/A.
- Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)? Currently, there are no plans for updating the dataset.
- If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced. Read the privacy policy for the NLU initiative at AI Sweden [here](https://www.ai.se/en/privacy-policy-nlu). 
- Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users. N/A.
- If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/ verified? If so, please describe how. If not, why not? Is there a process for communicating/ distributing these contributions to other users? If so, please provide a description. Not at this time.
- Any other comments? No.
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_AI-Sweden-Models__gpt-sw3-6.7b-v2-instruct)

| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 39.57   |
| ARC (25-shot)         | 40.78          |
| HellaSwag (10-shot)   | 67.77    |
| MMLU (5-shot)         | 31.57         |
| TruthfulQA (0-shot)   | 40.32   |
| Winogrande (5-shot)   | 63.54   |
| GSM8K (5-shot)        | 6.37        |
| DROP (3-shot)         | 26.67         |

---
language:
- sv
- da
- 'no'
license: llama3
base_model: AI-Sweden-Models/Llama-3-8B
---

# Checkpoint 1

## Training setup
The training was perfomed on the [LUMI supercomputer](https://lumi-supercomputer.eu/) within the [DeployAI EU project](https://www.ai.se/en/project/deployai). 
Based of the base model [AI-Sweden-Models/Llama-3-8B](https://huggingface.co/AI-Sweden-Models/Llama-3-8B).

## How to use

```python
import transformers
import torch

model_id = "AI-Sweden-Models/Llama-3-8B-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "Du är en hjälpsam assistant som svarar klokt och vänligt."},
    {"role": "user", "content": "Hur gör man pannkakor? Och vad behöver man handla? Undrar också vad 5+6 är.."},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])
```
```python
>>> "För att göra pannkakor behöver du följande ingredienser:
- 1 kopp vetemjöl
- 1 tesked bakpulver
- 1/4 tesked salt
- 1 kopp mjölk
- 1 stort ägg
- 2 matskedar smält smör eller olja

För att börja, blanda vetemjölet, bakpulvret och saltet i en bunke. I en annan skål, vispa ihop mjölken, ägget och smöret eller oljan. 
Tillsätt de våta ingredienserna till de torra ingredienserna och blanda tills det är väl blandat. 
Låt smeten vila i cirka 10 minuter.

För att göra pannkakorna, värm en non-stick-panna eller stekpanna över medelvärme. 
När den är varm, häll smeten på pannan och grädda tills kanterna börjar torka ut och toppen är fast. 
Vänd pannkakan med en stekspade och grädda den andra sidan tills den är gyllenbrun. 
Upprepa med resten av smeten.

När det gäller 5+6 är svaret 11."
```
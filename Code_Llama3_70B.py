
#import library
from airllm import AutoModel

#Length of the sequence
MAX_LENGTH = 128

#Import Llama3 70B
model = AutoModel.from_pretrained("v2ray/Llama-3-70B")
input_text = [        
  'What is the best AI company today?'    
]

#Tokenizer
input_tokens = model.tokenizer(input_text,    
  return_tensors="pt",     
  return_attention_mask=False,     
  truncation=True,     
  max_length=MAX_LENGTH,     
  padding=False)

generation_output = model.generate(    
  input_tokens['input_ids'].cuda(),     
  max_new_tokens=20,    
  use_cache=True,    
  return_dict_in_generate=True)

#Output
output = model.tokenizer.decode(generation_output.sequences[0])
print(output)

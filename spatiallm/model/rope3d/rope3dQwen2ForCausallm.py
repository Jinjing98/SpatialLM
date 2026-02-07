# inherit from Qwen2ForCausalLM
from transformers import Qwen2ForCausalLM
from transformers import Qwen2Model
class Qwen2ForCausalLMMixedRoPE3D(Qwen2ForCausalLM):
    pass
class Qwen2ModelMixedRoPE3D(Qwen2Model):
    pass

if __name__ == "__main__":

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
 
    # Option 2: Use HuggingFace model ID (requires internet)
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Step 1: Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Step 2: Load model with config and dtype (similar to SpatialLM inference)
    model = Rope3DQwen2ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,  # or torch.float16 for faster inference
        trust_remote_code=True
    )
 
    model.to("cuda")
    
    # Step 3: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        cache_implementation="static",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
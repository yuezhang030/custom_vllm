from vllm import LLM, SamplingParams

from huggingface_hub import login
login(token="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")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "DeepSpeed is a",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

llm = LLM(model="/mnt/output/youki/phi3-moe/hero-vanilla-phase2/123000_hf_bfloat16",
          tokenizer="/mnt/output/youki/phi3-moe/hero-vanilla-phase2/123000_hf_bfloat16",
        #   trust_remote_code=True,
          dtype="bfloat16", tensor_parallel_size=2, max_model_len=200)
# llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="mistralai/Mixtral-8x7B-v0.1", tensor_parallel_size=2, max_model_len=200)


outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


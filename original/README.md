<h3 align="center">"This is not a Disimprovement: Improving Negation Reasoning in Large
Language Models via Prompt Engineering"</h3>
    
<p align="justify">
We analyze various LLMs negation reasoning capabilities and propose two genres of prompts (Warning-based and Persona-based). We then ntroduce a negative token attention score (NTAS) to quantify attention to negation words.
</p>


<p align="center">
<img src="https://github.com/Joshua-Dias-Barreto/This-is-not-a-Disimprovement/blob/main/assets/example.png?raw=true" style="height: 450px;">
</p>


## Requirements
The scorer `evaluate.py` does not require any dependency. If you want to run the training or evaluation code you need:
```bash
# Required dependencies
Pytorch>=1.9 (2.1.0 Recommeneded) 
https://pytorch.org/get-started/locally/

transformers
pip install transformers

accelerate 
pip install accelerate

wandb
pip install wandb

# Optional dependencies

bitsandbytes >= 0.40.0 # For 4 / 8 bit quantization
pip install bitsandbytes

PEFT >= 0.4.0 # For LoRA
pip install peft

# You can install all the dependencies with:
pip3 install --upgrade torch transformers accelerate wandb bitsandbytes peft 
```

## Evaluating a LLM

We provide a script to evaluate any LLM in the dataset. First, you need to create a configuration file. 
See [configs/zero-shot](configs/zero-shot) for examples. This script will evaluate the model in our dataset in zero-shot setting.
Here is an example config to evaluate Qwen1.5-14b Chat:

> We will format the inputs using the chat_template stored in the tokenizer 

```yaml
#Model args
model_name_or_path: Qwen/Qwen1.5-14B-Chat
# Dtype in which we will load the model. You can use bfloat16 is you want to save memory
torch_dtype: "auto"
# Performs quatization using bitsandbytes integration. Allows evaluating LLMs in consumer hardware
quantization: 4
# If force_auto_device_map is set to True. We will split the model into all the available GPUs and CPU, this is useful for large models that do not fit in a single GPU VRAM. 
force_auto_device_map: false
# If set to false, we will sample the probability of generating the True or False tokens (recommended). If set to true, the model will generate a text and we will attempt to locate the string "true" or "false" in the output.
predict_with_generate: false
# Batch size for evaluation. We use auto_batch_finder, so this value is only used to set the maximum batch size, if the batch does not fit in memory, it will be reduced.
per_device_eval_batch_size: 32

# dataset arguments
do_train: false
do_eval: false
do_predict: true
# For zero-shot settings, you can evaluate the model in the concatenation of the train, dev and test sets. Set to false to only evaluate in the test set.
do_predict_full_dataset: false
max_seq_length: 4096

# Sampling parameters
sample_dataset: false
sample_size: 100

# Evaluate NTAS
calculate_nas: true

# Output Dir
output_dir: results/zero-shot/Qwen1.5-14B-Chat_original
````

Once you have created the config file, you can run the evaluation script:

```bash
accelerate launch run.py --config configs/zero-shot/Qwen1.5-14B-Chat.yaml
```

You can use accelerate to run the evaluation in multiple GPUs. See [accelerate documentation](https://github.com/huggingface/accelerate) for more information.
```bash
accelerate launch --multi_gpu --num_processes 2 run.py configs/zero-shot/Llama2-7b.yaml
```

If you want to evaluate multiple models, you can overwrite the `model_name_or_path` and `output_dir` values of a config. 

```bash
for model_name in \
meta-llama/Llama-2-70b-chat-hf \
meta-llama/Llama-2-70b-hf \
meta-llama/Llama-2-13b-chat-hf \
meta-llama/Llama-2-13b-hf \
meta-llama/Llama-2-7b-chat-hf \
meta-llama/Llama-2-7b-hf \
mistralai/Mistral-7B-Instruct-v0.2 \
mistralai/Mixtral-8x7B-Instruct-v0.1 \
do

accelerate launch run.py --config configs/zero-shot/base.yaml --model_name_or_path "$model_name" --output_dir results/zero-shot/"$model_name"

done
```

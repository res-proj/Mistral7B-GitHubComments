import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_from_disk
from trl import SFTTrainer
import pandas as pd
import time, os, json, gc, sys
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
# import huggingface_hub

# change to your Huggingface token
hf_token = "YOUR_HF_TOKEN"

# if device do not have access to internet, can use local cache
# download the cahce in another device, copy it to cache folder
USE_LOCAL_CACHE=False

# project's root path
base_path='./'

def create_prompt(sample):
    comment = sample["comment"].strip()
    theme = sample["Theme"].strip()
    instruction = "Assign a most suitable single theme to the given comment. The Input is the comment, give the theme in the Output."
    sys_prompt = "Below is an instruction that describe a task."
    prompt = f"<s>[INST] {sys_prompt}\n### Instruction:\n{instruction}\n### Input:\n{comment} [/INST] \n### Output:\n{theme} </s>"

    return prompt

def sorted_directory_listing_by_creation_time_with_os_listdir(directory):
    def get_creation_time(item):
        item_path = os.path.join(directory, item)
        return os.path.getctime(item_path)

    items = os.listdir(directory)
    sorted_items = sorted(items, key=get_creation_time, reverse=True)
    return sorted_items

def test_model(new_model, base_model, cache_dir, dataset):
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= False,
    )
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache = True,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )

    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=max_seq_length,
        trust_remote_code=True,
        add_bos_token=True,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )

    print("loading adapter")
    ft_model = PeftModel.from_pretrained(model, new_model)

    print(f"{new_model} testing......")
    result_list = []
    t_len = dataset['test'].num_rows
    for i in range(0, t_len):
        comment = dataset['test']['comment'][i]
        instruction = "Assign a most suitable single theme to the given comment. The Input is the comment, give the theme in the Output."
        sys_prompt = "Below is an instruction that describe a task."
        prompt = f"{sys_prompt}\n### Instruction:\n{instruction}\n### Input:\n{comment}\n### Output:\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        ft_model.eval()
        with torch.no_grad():
            generated = ft_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            o = output.replace(prompt, '').replace('[INST]', '').replace('[/INST]', '').replace('<s>', '').replace('</s>', '').strip()
            found = False
            for e in ["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"]:
                if e.lower() in o.lower():
                    found = True
                    result_list.append(e)
                    break
            if not found:
                result_list.append(o)
                print(f'{i}:{o}')
    print()
    print('accuracy')
    acc = accuracy_score(dataset['test']['Theme'], result_list)
    print(f'{acc}')
    print()
    print('precision, recall, f1 score')
    print('individual label')
    res = precision_recall_fscore_support(dataset['test']['Theme'], result_list, labels=["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"], average=None)
    print(f'{res}')
    print()
    print('micro')
    res = precision_recall_fscore_support(dataset['test']['Theme'], result_list, labels=["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"], average="micro")
    print(f'{res}')
    print()
    print('macro')
    res = precision_recall_fscore_support(dataset['test']['Theme'], result_list, labels=["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"], average="macro")
    print(f'{res}')
    print()
    print('weighted')
    res = precision_recall_fscore_support(dataset['test']['Theme'], result_list, labels=["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"], average="weighted")
    print(f'{res}')
    print()
    print(f"{new_model} test finished......")
    print()
    print()


if __name__ == "__main__":
    cfg_idx = int(sys.argv[1])

    # huggingface_hub.login(token="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")

    rs = [8, 16]
    drop_outs = [0.05, 0.1, 0]
    batch_sizes = [2, 4]
    lrs = [2.5e-5, 2e-4, 5e-5, 1e-5]
    
    r = rs[int(cfg_idx / 24)]
    drop_out = drop_outs[int(cfg_idx / 8) % 3]
    bs = batch_sizes[int(cfg_idx / 4) % 2]
    lr = lrs[int(cfg_idx % 4)]

    max_seq_length = 2048

    dataset = load_from_disk(base_path + 'dataset/dataset.hf')
    print(dataset)

    apdx = f'_{cfg_idx}_{r}_{drop_out}_{bs}_{lr}'
    print(apdx)

    cache_dir = base_path + 'cache'
    result = base_path + "result/result" + apdx
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    print(torch.cuda.is_available())
    print(device_map)

    print("load model")
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=False,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=max_seq_length,
        padding_side="left",
        add_eos_token=True,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("prepare LoRA")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=r,
        lora_alpha=16,
        lora_dropout=drop_out,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "lm_head"],
    )
    model = get_peft_model(model, peft_config)

    print("prepare training argument")
    # Training Arguments
    # Hyperparameters should beadjusted based on the hardware you using
    training_arguments = TrainingArguments(
        output_dir= result,
        num_train_epochs=6,
        per_device_train_batch_size= bs,
        weight_decay= 0.001,
        learning_rate=lr,
        lr_scheduler_type= "constant",
        gradient_accumulation_steps= 4,
        # warmup_steps = 10,
        warmup_ratio= 0.3,
        optim = "paged_adamw_8bit",
        save_strategy = "steps",
        save_steps= 10,
        eval_strategy="steps",
        eval_steps=1,
        logging_steps= 1,
        fp16 = True,
        bf16 = False,
        report_to="none",
        seed = 3407,
    )

    print("prepare trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        formatting_func = create_prompt,
        max_seq_length= max_seq_length,
        args=training_arguments,
        packing= True,
    )

    # save the fine-tuned model
    new_model = base_path + "jc/gh_comment_mistral_7b" + apdx
    # new_tokenizer = base_path + 'jc/gh_comment_mistral_7b' + apdx + '_tokenizer'
    print(f"{new_model} training......")
    trainer.train()
    trainer.model.save_pretrained(new_model, save_embedding_layers=True)
    # tokenizer.save_pretrained(new_model)

    print(f"{new_model} saving log")
    # save log
    with open(base_path + f'log/gh_comment_mistral_7b{apdx}.log', 'w+') as f:
        json.dump(trainer.state.log_history, f)

    # clean up
    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # test model
    test_model(new_model, base_model, cache_dir, dataset)

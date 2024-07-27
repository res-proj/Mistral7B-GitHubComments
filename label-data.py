import pandas as pd
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

# change to your token
hf_token = "YOUR_TOKEN"

# if device do not have access to internet, can use local cache
# download the cahce in another device, copy it to cache folder
# after training/fine-tuning, the cache is already downloaded
USE_LOCAL_CACHE=True

# project's root path
base_path='./'

def get_theme(ft_model, index, comment):
    instruction = "Assign a most suitable single theme to the given comment. The Input is the comment, give the theme in the Output."
    sys_prompt = "Below is an instruction that describe a task."
    prompt = f"{sys_prompt}\n### Instruction:\n{instruction}\n### Input:\n{comment}\n### Output:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = ft_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        o = output.replace(prompt, '').replace('[INST]', '').replace('[/INST]', '').replace('<s>', '').replace('</s>', '').strip()
        # print(f'{i}:{o}')
        for e in ["Improvement", "Project Workflow", "Interdependence", "Conflict", "Inquiry", "Unknow"]:
            if e.lower() in o.lower():
                return e
        # if generated output is not in the given themes, return Unknow
        print(f"{index}:{o}")
        return 'Unknow'

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

    base_path='/home/jcchen/projects/def-gerope/jcchen/gemini/'
    data_file=base_path + 'Gemini-Qualitative Analysis.xlsx'

    apdx = f'_{cfg_idx}_{r}_{drop_out}_{bs}_{lr}'

    cache_dir = base_path + 'cache'
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    lora_adaptor = base_path + f"result/result{apdx}/checkpoint-150"
    # lora_adaptor = f'{base_path}jc/gh_comment_mistral_7b{apdx}'

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= False,
    )
    print("load model")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )
    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        add_bos_token=True,
        trust_remote_code=True,
        token=hf_token,
        cache_dir=cache_dir,
        local_files_only=USE_LOCAL_CACHE,
    )

    print("load adapter")
    ft_model = PeftModel.from_pretrained(model, lora_adaptor)
    ft_model.eval()

    s_df = pd.read_excel(base_path + 'Gemini-Qualitative Analysis.xlsx', sheet_name='strong_discussions')
    w_df = pd.read_excel(base_path + 'Gemini-Qualitative Analysis.xlsx', sheet_name='weak_discussions')

    s_df.loc[s_df['Theme'] == '?', 'Theme'] = 'Unknow'
    w_df.loc[w_df['Theme'] == '?', 'Theme'] = 'Unknow'

    print('start labelling strong discussions')
    for idx, row in s_df.iterrows():
        if row['new'] <= 's0000000177':
            continue
        s_df.loc[idx, 'Theme'] = get_theme(ft_model, idx, row['comment'])
    print('save labelled strong discussions')
    s_df.to_csv(base_path + f'/strong_discussions{apdx}.csv', mode='w')

    print('start labelling weak discussions')
    for idx, row in w_df.iterrows():
        if row['new'] <= 'w0000000177':
            continue
        w_df.loc[idx, 'Theme'] = get_theme(ft_model, idx, row['comment'])
    print('save labelled weak discussions')
    w_df.to_csv(base_path + f'/weak_discussions{apdx}.csv', mode='w')
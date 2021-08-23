# DialoGPT
- [scrapbox](https://scrapbox.io/tohoku-nlp/Zhang+'20_DialoGPT:_Large-Scale_Generative_Pre-training_for_Conversational_Response_Generation_(ACL_2020))

## Set up

```bash
$ git clone git@github.com:smiyawaki0820/DialoGPT.git
$ cd DialoGPT
$ conda create -n dialogpt python=3.8 -y
$ pyenv local miniconda3-3.19.0/envs/dialogpt

$ qrsh 

$ module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 gcc/7.4.0
$ pip install -r requirements.txt

$ git clone https://github.com/YujiaBao/pytorch-pretrained-BERT.git
$ cd pytorch-pretrained-BERT
$ pip install -e .

$ git clone https://github.com/NVIDIA/apex _apex
$ cd _apex
$ rm -rf .git/
$ chmod -R 777 .
$ pip install torch==1.8.0
$ python setup.py install --cpp_ext --cuda_ext 
$ cd ..
```

```bash
# python demo.py --data dummy
```

## Datasets

```bash
# head data/train_raw.tsv
<context>\t<response>

# data/prepare4db.sh
# less data/train_raw.tsv | awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}'> data/train.tsv

$ paste -d "\t" <context> <response> | awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}' > data/train.tsv

# create ... data/train.128len.db
$ python prepro.py --corpus data/train.tsv --max_seq_len 128 --ja
```

## Models

### English:

```python
>>> from functools import partial
>>> from demo_utils import download_model_folder
>>> download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)
>>> target_folder = download_model(model_size='small', dataset='multiref', from_scratch=False)
```

```bash
$ ls models/small
pytorch_model.bin config.json vocab.json small_ft.pkl merges.txt

$ cat models/small/config.json
{
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 1024,
  "vocab_size": 50257
}
```

### Japanese
- [Question: Does it only work for English? #7](https://github.com/microsoft/DialoGPT/issues/7)
- [Japanese GPT-2](https://huggingface.co/models?search=gpt2+ja)
  - [rinna/japanese-gpt2-medium](https://huggingface.co/rinna/japanese-gpt2-medium)

```python
>>> from transformers import T5Tokenizer, AutoModelForCausalLM

>>> tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
>>> tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
>>> model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

>>> model.config
GPT2Config {
  "_name_or_path": "rinna/japanese-gpt2-medium",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 1,
  "embd_pdrop": 0.1,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_inner": 4096,
  "n_layer": 24,
  "n_positions": 1024,
  "resid_pdrop": 0.1,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.9.2",
  "use_cache": true,
  "vocab_size": 32000
}
```


## Train

```bash
bash train.sh
```

```bash
$ MODEL_OUT=models/outputs
$ TRAIN_FILE=data/train.128len.db
$ EVAL_FILE=data/dummy_data.tsv

# Training on 8 GPUs
# python -m torch.distributed.launch --nproc_per_node=8 ./LSP_train.py
$ python LSP_train_ja.py \
  --model_name_or_path rinna/japanese-gpt2-medium \
  --init_checkpoint "None" \
  --train_input_file $TRAIN_FILE \
  --eval_input_file $EVAL_FILE \   # dummy test data
  --output_dir $MODEL_DIR \
  --seed 42 \
  --max_seq_length 128 \
  --train_batch_size 512 \
  --gradient_accumulation_steps 8 \
  --eval_batch_size 64 \
  --learning_rate 1e-5 \
  --num_optim_steps 10000 \
  --valid_step 5000 \
  --warmup_steps 4000 \
  --normalize_data true \
  --fp16 true \
  --lr_schedule noam \
  --loss_scale 0.0 \
  --no_token_id true \
  --pbar true \
```

import json
import time
from pprint import pprint

with open('/scratch/gilbreth/dchawra/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json', 'r') as file:
    data = json.load(file)

new_data = []
for entity in data:
    if len(entity['conversations']) > 0:
        new_data.append(entity)
    # elif len(entity['conversations']) == 1:
    #     print(entity['conversations'])
    else:
        print('No conversations')

with open('/scratch/gilbreth/dchawra/sgpt_nonempty.json', 'w') as file:
    json.dump(new_data, file, indent=4)


torchrun --nproc_per_node=3 --master_port=20001 train/train_mem.py --model_name_or_path /scratch/gilbreth/dchawra/vicuna-7b  --data_path /scratch/gilbreth/dchawra/sgpt_nonempty.json --bf16 True --output_dir /scratch/gilbreth/dchawra/vicuna-ft --num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 10 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap offload" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' --tf32 True --model_max_length 2048 --gradient_checkpointing True --lazy_preprocess True
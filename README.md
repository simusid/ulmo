# ulmo
Tools and resources to build and test a Large Acoustics Model (LAM) 

# Wav2Vec2
https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20
Goal is to duplicate training the pretrained model using non-speech data.

## Running Wav2Vec2

Pull huggingface/transformer repo.  Then cd to examples/pytorch/speech-pretraining.

Then:

export PYTORCH_ENABLE_MPS_FALLBACK=1

and 

accelerate launch run_wav2vec2_pretraining_no_trainer.py \
        --dataset_name="librispeech_asr" \
        --dataset_config_names clean clean \
        --dataset_split_names validation test \
        --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
        --output_dir="./wav2vec2-pretrained-demo" \
        --max_train_steps="20000" \
        --num_warmup_steps="32000" \
        --gradient_accumulation_steps="8" \
        --learning_rate="0.005" \
        --weight_decay="0.01" \
        --max_duration_in_seconds="20.0" \
        --min_duration_in_seconds="2.0" \
        --logging_steps="1" \
        --saving_steps="10000" \
        --per_device_train_batch_size="8" \
        --per_device_eval_batch_size="8" \
        --adam_beta1="0.9" \
        --adam_beta2="0.98" \
        --adam_epsilon="1e-06" \
        --gradient_checkpointing \
        --mask_time_prob="0.65" \
        --mask_time_length="10"


if this runs, the next goal is to replace the librispeech_asr named datasource and replace it with other
acoustic wav files.  This will probably use the AudioFolder builder

# Whisper
https://arxiv.org/pdf/2212.04356.pdf


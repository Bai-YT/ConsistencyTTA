batch_size=32
which_gpu=0
num_steps=1

ref_dir="data/audiocaps_test_references/subset"
ref_json="data/test_audiocaps_subset.json"

for epoch in "10"  # 60 for ConsitencyTTA, 10 for ConsistencyTTA_CLAPFT
do
    for guidance in 4 5
    do
        # ConsistencyTTA model
        CUDA_VISIBLE_DEVICES=$which_gpu python inference.py \
        --original_args="saved/ConsistencyTTA/summary.jsonl" \
        --model="saved/ConsistencyTTA/epoch_"$epoch"/pytorch_model_2.bin" \
        --test_file=$ref_json --test_references=$ref_dir --seed=0 \
        --stage=2 --guidance_scale_input=$guidance --guidance_scale_post=1 \
        --num_steps=$num_steps --batch_size=$batch_size --use_edm --use_ema --use_bf16

        # ConsistencyTTA model after CLAP fine-tuning
        CUDA_VISIBLE_DEVICES=$which_gpu python inference.py \
        --original_args="saved/ConsistencyTTA_CLAPFT/summary.jsonl" \
        --model="saved/ConsistencyTTA_CLAPFT/epoch_"$epoch"/pytorch_model_2.bin" \
        --test_file=$ref_json --test_references=$ref_dir --seed=0 \
        --stage=2 --guidance_scale_input=$guidance --guidance_scale_post=1 \
        --num_steps=$num_steps --batch_size=$batch_size --use_edm --use_ema --use_bf16
    done
done

# Evaluate existing generated audio clips
# python evaluate_from_existing.py \
#     --gen_dir ../tango/data/audiocaps_test_references/subset

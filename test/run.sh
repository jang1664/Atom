python test_zero_shot.py ../llama2-7b wikitext2 \
        --wbits 4 --abits 4 --a_sym --w_sym --save_model \
        --act_group_size 128 --weight_group_size 128 --weight_channel_group 2 \
        --reorder --act_sort_metric hessian --cache_index \
        --a_clip_ratio 0.9 --w_clip_ratio 0.85 --kv_clip_ratio 1.0 \
        --keeper 128 --keeper_precision 3 --kv_cache --use_gptq \
        --eval_common_sense --lm_eval_limit -1 \
        --save_dir ../saved \
        --save_model_name llama7b_quant_quantized.pth \
        --test_name llamma2-7b-quant
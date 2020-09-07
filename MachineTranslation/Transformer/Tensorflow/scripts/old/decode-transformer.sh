mkdir -p ../t2t_decodes
python "../source/tensor2tensor/tensor2tensor/bin/t2t-decoder.py" --data_dir=../t2t_data --problems=translate_ende_wmt32k --model=transformer --hparams_set=transformer_base_single_gpu --output_dir="../t2t_averaged/" --decode_hparams="beam_size=5,alpha=0.6" --decode_to_file="../t2t_decodes/results.txt" --decode_hparams='use_last_position_only=True'


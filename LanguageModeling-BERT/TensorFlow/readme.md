# Language Modeling with BERT

## Introduction

This folder contains the TensorFlow implementation of the BERT model for language modeling.  

The dataset directory contains:
    - model: pretrained weights and config file for the base uncased english model
    - classification: GLUE data for the classification task
    - pretrain: Wikipedia data for pretraining BERT from scratch 

The scripts directory contains:
    - pretrain: 
        - create_pretrain_record.py: script from the NVIDIA/DeepLearningExamples BERT repo that transforms text files to tfrecord files.  
        - format_wiki_data.py: script from the NVIDIA/DeepLearningExamples BERT repo that formats the output of the WikiExtractor.
        - TextSharding.py: script from the NVIDIA/DeepLearningExamples BERT repo that shards a formatted text file.
        - WikiExtractor.py: WikiExtractor tool that extracts Wiki data dump into a cleaner xml format. 
        - pretrain.sh: initiates multi gpu mixed-precision pretraining of BERT.
    - download_classification_data.sh: runs download_glue_data.py to download data for classification tasks.
    - download_glue_data.py: script by W4ngatang that downloads and extracts GLUE data.
    - download_model.sh: downloads pretrained weights and config files for BERT.
    - profile_nsigh.sh: profiles BERT classification fine tuning with nsight compute.
    - profile_nvprof.sh: profiles BERT classification fine tuning with nvprof. 

The source directory contains:
    - TensorFlow implementation of BERT by Google.
    - optimization_custom.py: updated version of optimization.py that uses the base TF Adam optimizer for distributed strategy compatibility.
    - run_pretraining_custom.py: updated version of run_pretraining.py that switches from TPUEstimator to Estimator, logs next sentence and masked lm loss to training via tensorboard, and uses data distributed and mixed-precision training. 

## References

[BERT TensorFlow Implementation] (https://github.com/google-research/bert)

[NVIDIA/DeepLearningExamples] (https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

[WikiExtractor] (https://github.com/attardi/wikiextractor)

[download_glue_data.py] (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)

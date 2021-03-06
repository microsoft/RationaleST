# Self-training with Few-shot Rationalization

Self-training with Few-shot Rationalization is a multi-task learning (MTL) based framework based on self-training language models with few labels for task and rationales.

Our paper was published at EMNLP 2021. The link to the paper can be found here: [Self-training with Few-shot Rationalization: Teacher Explanations Aid Student in Few-shot NLU](https://arxiv.org/pdf/2109.08259.pdf)

## Overview

![screenshot](screenshot.png)

## Using the Code

### Installing the dependencies

To run the code, please install the packages from ```requirements.txt``` using the command: ``` pip install -r requirements.txt```

### Downloading the datasets

Datasets used in this paper are from ERASER benchmark. The datasets can be downloaded from this [link](https://www.eraserbenchmark.com)

### Training our model

To run the code:
```
python run_st_rationale.py --task <dataset_dir> --model_dir <output_dir>  --seq_len 512 --sample_scheme uniform --sup_labels 200 --valid_split 0.2 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-uncased --N_base 3 --sup_batch_size 8 --sup_epochs 70 --unsup_epochs 20 --model_type joint_neg_rwt_l_r_fine_tune_teacher 
```
*Classification tasks*: Set --do_pairwise for evidence and boolq dataset

*Model type*: Option *model_type*: ```joint_neg``` (MTL framework without reweighting), ```joint_neg_rwt_l_r``` (MTL framework with reweighting), ```joint_neg_rwt_l_r_fine_tune_teacher``` (MTL framework with reweighting and fine-tuning teacher)

*Training and validation*: ```sup_labels``` denote the total number of available labeled examples for each class, where ```valid_split``` uses a fraction of those labels as validation set for early stopping. Set ```sup_labels``` to -1 to use all training labels. Set ```valid_split``` to -1 to use the available test data as the validation set.

*Fine-tuning batch size*: Set ```sup_batch_size``` to a small number for few-shot fine-tuning of the teacher model. In case of many training labels, set ```sup_batch_size``` to a higher value for faster training.

*Task name*: Option ```task``` Provide the inout dataset directory for training.

*HuggingFace Transformers*: To use different pre-trained language models from HuggingFace, set pt_teacher and pt_teacher_checkpoint to corresponding model versions available from [here](https://huggingface.co/transformers/pretrained_models.html). A default set of pre-trained language models is available at ```huggingface_utils.py```.

### Citation

If you find our paper useful, please cite the following:

```
@InProceedings{bhat2021self-training,
author = {Bhat, Meghana Moorthy and Sordoni, Alessandro and Mukherjee, Subhabrata (Subho)},
title = {Self-training with Few-shot Rationalization: Teacher Explanations Aid Student in Few-shot NLU},
booktitle = {EMNLP 2021},
year = {2021},
month = {November},
abstract = {While pre-trained language models have obtained state-of-the-art performance for several natural language understanding tasks, they are quite opaque in terms of their decision-making process. While some recent works focus on rationalizing neural predictions by highlighting salient concepts in text as justifications or rationales, they rely on thousands of labeled training examples for both task labels as well as annotated rationales for every instance. Such extensive large-scale annotations are infeasible to obtain for many tasks. To this end, we develop a multi-task teacher-student framework based on self-training pre-trained language models with limited task-specific labels and rationales and judicious sample selection to learn from informative pseudo-labeled examples. We study several characteristics of what constitutes a good rationale and demonstrate that the neural model performance can be significantly improved by making it aware of its rationalized predictions particularly in low-resource settings. Extensive experiments in several benchmark datasets demonstrate the effectiveness of our approach.},
url = {https://www.microsoft.com/en-us/research/publication/self-training-with-few-shot-rationalization/},
}
```

### Contact

For questions, please contact [Meghana Bhat](https://meghu2791.github.io)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

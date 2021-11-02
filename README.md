# Self-training with Few-shot Rationalization

Self-training with Few-shot Rationalization is a multi-task learning based framework based on self-training language models with few labels for task and rationales.

Our paper was published at EMNLP 2021. The link to the paper can be found here: [Self-training with Few-shot Rationalization: Teacher Explanations Aid Student in Few-shot NLU](https://arxiv.org/pdf/2109.08259.pdf)

## Using the code


To run the code:
```
python run_st_rationale.py --task ../data/<dataset> --model_dir <output_dir>  --seq_len 512 --sample_scheme uniform --sup_labels 200 --valid_split 0.5 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-uncased --N_base 3 --sup_batch_size 4 --sup_epochs 100 --unsup_epochs 20 --model_type joint_neg_rwt_l_r_fine_tune_teacher 
```
Add --do_pairwise for evidence and boolq dataset


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

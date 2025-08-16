# StereotypeAsCatalystForBias
Dataset can also be viewed at [StereoBias](https://huggingface.co/datasets/aditya20t/StereoBias)

This repository accompanies the paper [Stereotype Detection as a Catalyst for Enhanced Bias Detection: A Multi-Task Learning Approach](https://aclanthology.org/2025.findings-acl.889/)

---
### Citation
```
@inproceedings{tomar-etal-2025-stereotype,
    title = "Stereotype Detection as a Catalyst for Enhanced Bias Detection: A Multi-Task Learning Approach",
    author = "Tomar, Aditya  and
      Murthy, Rudra  and
      Bhattacharyya, Pushpak",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.889/",
    doi = "10.18653/v1/2025.findings-acl.889",
    pages = "17304--17317",
    ISBN = "979-8-89176-256-5",
    abstract = "Bias and stereotypes in language models can cause harm, especially in sensitive areas like content moderation and decision-making. This paper addresses bias and stereotype detection by exploring how jointly learning these tasks enhances model performance. We introduce StereoBias, a unique dataset labeled for bias and stereotype detection across five categories: religion, gender, socio-economic status, race, profession, and others, enabling a deeper study of their relationship. Our experiments compare encoder-only models and fine-tuned decoder-only models using QLoRA. While encoder-only models perform well, decoder-only models also show competitive results. Crucially, joint training on bias and stereotype detection significantly improves bias detection compared to training them separately. Additional experiments with sentiment analysis confirm that the improvements stem from the connection between bias and stereotypes, not multi-task learning alone. These findings highlight the value of leveraging stereotype information to build fairer and more effective AI systems."
}
```

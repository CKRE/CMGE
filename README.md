# CMGE
## DataType
Due to privacy issues in the medical record data, we can only provide the format of the data. The text is only for reference.
```
sample_id<->1 # The ID of this medical record.
type<->2 # The diagnosis of this medical record.
diagnose<->胸部存在磨玻璃阴影 # Description of specific diagnostic results.
age<->31
gender<->男
desc1<->患者8星期前胸片检查发现磨玻璃影。 # The text of the first clause.
pattern1<->患者<时长>前胸片检查发现<疾病>。 # The entity recognition results of the first clause.
semantic1<->诊疗经过_检查及诊断结果 # The clause type of the first clause.
desc2<->未予特殊治疗
pattern2<->未予特殊治疗
semantic2<->诊疗经过_治疗手段
...
```
## Citation
```bibtex
@inproceedings{wu-etal-2021-counterfactual,
    title = "Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network",
    author = "Wu, Haoran  and
      Chen, Wei  and
      Xu, Shuang  and
      Xu, Bo",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.156",
    doi = "10.18653/v1/2021.naacl-main.156",
    pages = "1942--1955",
}
```

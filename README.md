# LitLLM
*Literal-Aware KGC (GOBLIN COST Action CA23147)*


This work is part of research **"Literal-Aware Refinement of Knowledge Graphs via Large Language Models"**  conducted during a **Short-Term Scientific Mission (STSM) 
under the GOBLIN COST Action (CA23147)**, hosted at the Department of Computer Science, 
Aalborg University, Denmark. 
*(manuscript in preparation)*

This repository contains part of the resulting implementation:


## Contents

- `train_base_retriever.py` – trains the baseline retrieval model
- `tune_retriever.py` – fine-tunes the retriever
- `run_tuning.py` – runs the tuning pipeline
- `tuning_config.yml` – configuration for tuning experiments
- `litwd_pykeen_loader.py` – data loader for the LiterallyWikidata benchmark
- `decode_candidates.py` – decodes candidate entities
- `export_from_tuned_model.py` – exports results from a tuned model
- `LiterallyWikidata-main/` – datasets (LitWD1K, LitWD19K, LitWD48K). The datasets are taken from  Gesese et al. (2023). *LiterallyWikidata*. https://github.com/GenetAsefa/LiterallyWikidata 

## Status

Work in progress. The repository is made public only for the STSM report evaluation process. A full release including the complete pipeline and results will be made available upon paper acceptance.
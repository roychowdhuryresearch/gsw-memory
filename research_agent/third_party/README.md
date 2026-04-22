# third_party/

Upstream competitor repos cloned here for adapter development. All contents of this directory are gitignored (see `../.gitignore`) except this README.

Planned clones:

| Dir | Upstream | Purpose |
|---|---|---|
| `Search-R1/` | [github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Inference path for the Search-R1 adapter |
| `Search-o1/` | [github.com/RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1) | Prompt-only framework wrapper |
| `ASearcher/` | [github.com/inclusionAI/ASearcher](https://github.com/inclusionAI/ASearcher) | Prompt-mode inference + trained-ckpt serving |
| `context-1-data-gen/` | [github.com/chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen) | Reference for Context-1 tool schema |
| `DeepResearch/` | [github.com/Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Tongyi DeepResearch reference implementation |

Clone with:
```bash
cd third_party
git clone --depth 1 https://github.com/PeterGriffinJin/Search-R1.git
git clone --depth 1 https://github.com/RUC-NLPIR/Search-o1.git
git clone --depth 1 https://github.com/inclusionAI/ASearcher.git
git clone --depth 1 https://github.com/chroma-core/context-1-data-gen.git
git clone --depth 1 https://github.com/Alibaba-NLP/DeepResearch.git
```

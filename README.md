# DevShinra2022

## 分類タスク

### 前処理

[ダウンロードページ](https://2022.shinra-project.info/data-download)から以下をダウンロードし、`./data`直下においてください。
- 教師データ（JSONL）: `ENE9-0-Wikipedia2019-all-20220511.json`
- Wikipedia2019 (CirrusSearchDump): `jawiki-20190121-cirrussearch-content.json.gz`

~~~bash
python3 src/preprocess.py \
    preprocess=classification
~~~

### 学習

~~~bash
python3 src/train.py \
    hydra.run.dir=outputs/classification \
    train=classification \
    train.model.bert.dir=models/roberta_large_wiki201221_janome_bpe_merge_10000_vocab_24000 \
    train.total_updates=30000 \
    train.gradient_accumulation_steps=2 \
    train.eval_updates=1000 \
    train.checkpoint_updates=1000 \
    train.dataloader.train.batch_size=128 \
    train.dataloader.eval.batch_size=256
~~~

### 予測

#### リーダーボード

[ダウンロードページ](https://2022.shinra-project.info/data-download)から以下をダウンロードし、`./data`直下においてください。
- リーダボード入力データ（JSONL）: `shinra2022_Categorization_leaderboard_20220530.jsonl`
- Wikipedia2021 (CirrusSearchDump) : `wikipedia-ja-20210823-json.gz` 


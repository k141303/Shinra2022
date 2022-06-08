# 森羅2022RoBERTaベースライン

日本語事前学習済みRoBERTaを使用して簡単なベースラインシステムを作成しました。  
現在、分類タスクのみ実装が完了しています。  
属性値抽出タスクにつきましては少々お待ちください。

## 学習済み予測結果
後続のタスクにご利用ください。
- [分類タスク](https://drive.google.com/drive/folders/1ZJOt3TlSB1soSxwRmShdHKQhZkkAWrXP?usp=sharing)

## 分類タスク

### 必要なデータ

[ダウンロードページ](https://2022.shinra-project.info/data-download)から以下をダウンロードし、`./data`直下に置いてください。
- 教師データ（JSONL）: `shinra2022_Categorization_train_20220511.jsonl`
- Wikipedia2019 (CirrusSearchDump): `jawiki-20190121-cirrussearch-content.json.gz`
- Wikipedia2021 (CirrusSearchDump) : `wikipedia-ja-20210823-json.gz` 
- リーダボード入力データ（JSONL）: `shinra2022_Categorization_leaderboard_20220530.jsonl`
- 本評価の入力データ（JSONL）: `shinra2022_Categorization_test_20220511.jsonl`

[GoogleDrive](https://drive.google.com/drive/folders/1e7QqjWDhbFkOyeIqJdcsK4whfBW5QWWY?usp=sharing)から事前学習済みRoBERTaをダウンロードし、解凍後`./models`直下に置いてください。  
- `roberta_large_wiki201221_janome_bpe_merge_10000_vocab_24000.zip`

本スクリプトはlargeモデルを使用していますが、計算コストを削減したい場合は小さいモデルであるbaseを使用できます。  
両者ともトークナイズや語彙は共通なため、いずれのモデルを指定して前処理を行ってもデータを使い回すことが可能です。
- `roberta_base_wiki201221_janome_bpe_merge_10000_vocab_24000.zip`

### 前処理

#### Wikipedia2019 (学習用)

~~~bash
python3 src/preprocess.py \
    preprocess=classification \
    preprocess.model.dir=models/roberta_large_wiki201221_janome_bpe_merge_10000_vocab_24000
~~~

#### Wikipedia2021 (予測用)

~~~bash
python3 src/preprocess.py \
    preprocess=classification \
    preprocess.model.dir=models/roberta_large_wiki201221_janome_bpe_merge_10000_vocab_24000 \
    preprocess.data.ene_name=null \
    preprocess.data.cirrus_name=wikipedia-ja-20210823-json.gz
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

~~~bash
python3 ./src/predict.py \
    hydra.run.dir=outputs/classification \
    predict=classification \
    predict.data.target_name=shinra2022_Categorization_leaderboard_20220530.jsonl
~~~

#### 本参加

~~~bash
python3 ./src/predict.py \
    hydra.run.dir=outputs/classification \
    predict=classification \
    predict.data.target_name=shinra2022_Categorization_test_20220511.jsonl
~~~

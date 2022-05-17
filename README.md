# DevShinra2022

## ENE分類

### 前処理

~~~bash
python3 src/preprocess.py \
    preprocess=classification
~~~

### 学習

~~~bash
src/train.py \
    hydra.run.dir=outputs/classification \
    train=classification \
    train.total_updates=30000 \
    train.gradient_accumulation_steps=4 \
    train.eval_updates=1000 \
    train.checkpoint_updates=1000 \
    train.dataloader.train.batch_size=64 \
    train.dataloader.eval.batch_size=256
~~~

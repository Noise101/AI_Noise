# Experiments

実験は番号順に、前の版の制約を一つずつ外す形で構成されています。最新版だけでなく、失敗や比較対象を再現できるよう各版を残しています。

```bash
python3 unified_agent_v9.py --trials 50
python3 -m unittest discover -v
```

主な評価スクリプト：

- `evaluate_v1.py`: 即時因果規則の複数seed評価
- `evaluate_v2.py`: 遅延探索あり／なしの比較
- `concept_transfer_v3.py`: 概念転移と負の転移
- `self_correcting_agent_v6.py`: 誤概念の自己撤回
- `active_falsification_v7.py`: 能動反証と受動観測の比較
- `constructive_concepts_v8.py`: 全仮説列挙なしの構造成長
- `unified_agent_v9.py`: 統合評価
- `probabilistic_agent_v10.py`: 確率的因果、ノイズ下の構造回収と校正
- `web_learning_v11.py`: 調査目標を生成する読み取り専用ウェブ学習
- `story_learning_v12.py`: 児童向け短文による予測、驚き、自己訂正、「なぜ？」の生成

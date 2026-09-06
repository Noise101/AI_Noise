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
- `story_web_curriculum_v13.py`: 不足から検索語を生成し、公開児童文学を複数資料から読むカリキュラム
- `story_concepts_v14.py`: 表現差を統合し、視点・出典・反証を保持する概念台帳
- `developmental_language_v15.py`: 文字、語、句、意味役割、文因果の並行学習
- `lexical_research_v16.py`: 自己生成した未知語検索、複数sense評価、語彙記憶への書き戻し
- `phrase_learning_v17.py`: 反復句の検索と構成的／非構成的意味の証拠評価
- `japanese_boundaries_v18.py`: 日本語児童文からの境界誘導、辞書・百科事典による検証
- `japanese_sense_grounding_v19.py`: 日本語多義語の候補列挙、文脈接地、反証訂正
- `autonomous_controller_v20.py`: 期待情報利得による課題選択、予算停止、状態・知識の永続化と再開
- `local_worker_v21.py`: Codex/APIを呼ばないローカル反復実行、heartbeat、再開、安全停止
- `kanjipedia_reference_v22.py`: 漢字ペディアの完全一致項目を本文転載なしで構造検証
- `curiosity_drive_v23.py`: 再遭遇・複数文脈・未解決時間で増える知りたい圧
- `mastery_drive_v24.py`: 言語能力の自己評価と最弱層からの次の習得目標
- `local_conversation_v25.py`: 自作した問いによるOllamaとの短い会話練習（証拠スコア0）
- `compact_runtime_v26.py`: seed固有知識を保った冗長台帳の除去とストレージ回収
- `global_memory_v27.py`: 全題材の語彙・会話・出来事・概念を一つの正本へ統合
- `narrative_event_v29.py`: 書誌・曖昧文を棄却理由付きで隔離する透明な出来事抽出
- `causal_lab_v30.py`: 操作可能な未知小世界で能動介入能力だけを検査（世界知識には加算しない）
- `developmental_curriculum_v32.py`: 現在能力への適合度で資料の記憶採用・延期・派生停止を決定
- `epistemic_scaffold_v34.py`: 人間科学の将来学習用に観測と未記入の解釈欄を分離保存
- `error_memory_v35.py`: 誤予測・反証・訂正内容を重複なく保存する横断的な誤り記憶
- `visual_memory_v36.py`: 出典付き縮小画像を視覚特徴として観測し、言語との未検証連想を保存
- `event_structure_v1.py`: world_model_v51／association_learning_v33／causal_experiment_v28／representation_learning_v31 を置換。次イベント予測に有意信号がないため、イベント内部構造（動詞クローズ・妥当性判定）を出典分離の固定ベンチマークで評価
- `coreference_v1.py`: 出典内共参照。代名詞・エンティティを数・有生性・近接窓で解決し、抽出前に主語スレッドを一本化
- `proposition_v1.py`: 行動抽出が捨てる繋辞・所有節を entity|relation|value 命題に変換（現状はデータ供給のみ）
- `sequence_model_v1.py`: ゼロから学習する極小文字RNN（隠れ24、numpyなし、手書きBPTT）。連続的 bits/char 信号と生成ヘッド
- `active_curriculum_v1.py`: 頻度駆動の閉じ級好奇心を退役させ、固定ベンチマークの誤りを検索シードに変える能動学習
- `capability_report_v1.py`: 連続・多次元の能力ダッシュボード（学習効率＝lift勾配、perplexity、抽出健全性、ゲート）
- `llm_tooluse_v1.py`: 検証可能なサブタスクをローカルモデルに投げ、接地語彙とパーサで検証し、有効なプロンプト形式を学習
- `generative_dialogue_v1.py`: 発話を構成しローカルモデルを環境として使い、伝達成功を報酬に構成戦略を学習

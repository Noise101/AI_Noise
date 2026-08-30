# AI_Noise

巨大LLMの縮小模倣ではなく、少ない経験から概念と因果関係を形成し、予測の失敗から自己訂正するAIアーキテクチャの実験プロジェクトです。

現在の研究対象は、次の循環です。

```text
観測前予測
  → 現実との誤差
  → 競合仮説の比較
  → 概念構造の追加・削除・置換
  → 反証に適した介入の選択
  → 再予測
```

LLMと学習済みモデルは使用していません。実験はPython標準ライブラリだけで動作します。v11だけは読み取り専用のWikimedia APIへ接続します。

## 現在地

`experiments/unified_agent_v9.py` が現在の統合版です。以下を単一の継続学習ループにまとめています。

- 結果を観測する前の事前予測
- 予測誤差による自己評価
- 真理値表による概念形成（AND・OR・XORの固定文法なし）
- 概念入力の追加・削除・置換
- 遅延因果の探索
- MDL（記述長）による自己改訂
- 競合モデルを分ける能動介入
- 非定常環境への再適応

50 seedの比較では、最終汎化精度が能動介入95.5%、受動観測79.0%、完全回復率が90%、56%でした。これは限定された二値の玩具世界での結果であり、一般知能の達成を意味しません。

## 実行

Python 3.11以降を想定しています。追加パッケージは不要です。

```bash
cd experiments
python3 unified_agent_v9.py --trials 50
python3 -m unittest discover -v
```

長い評価は、完了seed数を逐次表示します。

## 実験の発展

| 版 | 主題 |
|---|---|
| v0 | 複数仮説・驚き・規則変更 |
| v1 | 観測からの規則候補生成 |
| v2 | 遅延因果と時間的信用割当 |
| v3 | 概念の圧縮と少数例転移 |
| v4 | 固定論理演算子なしのXOR概念形成 |
| v5 | 遅延因果と概念転移の統合 |
| v6 | 予測誤差から誤概念を自律撤回 |
| v7 | 自分を反証する実験の能動選択 |
| v8 | 残差から概念構造を逐次成長 |
| v9 | 上記機能を単一ループへ統合 |
| v10 | 確率的因果・観測ノイズ・信念校正 |
| v11 | 調査目標生成と読み取り専用ウェブ学習 |
| v12 | 児童向け短文からの出来事予測・自己訂正・「なぜ？」生成 |
| v13 | 知識不足から公開児童文学を自律検索する読み取り専用カリキュラム |
| v14 | 資料間の表現統合・視点分離・反証による概念信念更新 |
| v15 | 文字・単語・句・意味役割・文因果の並行発達学習 |
| v16 | 未知語の辞書・児童文用例調査と出典付きsense書き戻し |

各版は`experiments/`内に独立した実行可能ファイルとして残しています。

## 次の課題

- 決定論的真理値表から確率的因果モデルへの移行
- 観測ノイズ下での信念校正
- 無関係センサーが多い環境での構造探索
- 概念の階層化と再利用
- 玩具世界以外へ接続できる環境インターフェース

## v10

`experiments/probabilistic_agent_v10.py` は、各因果状態を0/1で確定せずBeta分布として保持します。確率的結果の下で、構造・遅延の探索、事前予測、Brier scoreによる校正評価、能動介入を行います。

```bash
cd experiments
python3 probabilistic_agent_v10.py --trials 50
```

## v11

`experiments/web_learning_v11.py` は、与えられた話題を起点にWikidataを検索し、未探索の関連エンティティから次の調査目標を自ら生成します。APIアクセスは読み取り専用です。この版は既成の知識グラフに依存しすぎ、複数値を矛盾と誤認するため、完成版ではなく失敗を含む比較対象です。

```bash
cd experiments
python3 web_learning_v11.py "causal inference" --steps 8 --output report.json
```

## v12

`experiments/story_learning_v12.py` は、児童文学・絵本相当の単純な出来事列から始めます。短文を透明な規則で出来事へ変換し、次の出来事を予測します。予想外の結果は誤りとして保存され、反例が重なると古い予測が置き換わります。

「なぜ？」は知識の暗唱ではありません。予想外の結果に対して自動生成され、ある文脈の後で結果が増えたかを他の文脈と比較します。比較証拠が足りなければ`unknown`、足りれば検証可能な候補原因として回答します。現段階では相関的な候補であり、原因の証明ではありません。

```bash
cd experiments
python3 story_learning_v12.py
python3 -m unittest -v test_story_learning_v12.py
```

## v13

`experiments/story_web_curriculum_v13.py` は、v12が持つ未解決のWhy質問または観察回数の少ない規則を知識不足として検出し、そこから検索語を生成します。固定の正解ページは持たず、WikisourceとProject Gutenbergを別々に読み取り専用検索します。

取得本文からライセンス文・書誌情報・別作品を除外し、透明な動作語規則で出来事を抽出します。各資料についてURL、検索語、取得本文と使用箇所のSHA-256、抽出イベントを証拠台帳へ保存します。2つの独立リポジトリから本文を取得できない場合、結論は`uncertain`のままです。

```bash
cd experiments
python3 story_web_curriculum_v13.py "fox grapes" --rounds 2 --output report.json
python3 -m unittest -v test_story_web_curriculum_v13.py
```

現段階の文法は英語の単純な行動文に限定されています。代名詞の同一人物判定や、文章に明記されない原因・意図の理解は未実装です。

## v14

`experiments/story_concepts_v14.py` は、`tried / tricks / jumped`など異なる表現を少数の検証可能な概念へ統合します。支持と反対を独立資料ごとに加重し、単一資料、複数資料一致、係争中、反証側を暫定採用、を区別します。語り手の事実と登場人物の信念は別scopeなので、「熟した葡萄」と狐の「酸っぱい」は誤って矛盾扱いしません。結論にはURLと不確実性を付けます。

## v15

`experiments/developmental_language_v15.py` は、一文を読むたびに以下を並行更新します。

- 文字の出現と隣接
- 空白で観察された英語語形と語の隣接
- 反復する句候補
- 出来事に接地したagent/action/object-or-detail役割
- 出来事の予測と資料横断概念
- 意味未確定語または日本語の境界未確定文字列からの次の検索語

日本語は事前tokenizerで分割せず、反復する2〜4文字列を境界候補として保持します。これは単語候補の発見であり、意味理解の完了ではありません。英語でも「見た語」と「役割が接地した語」を区別します。

```bash
cd experiments
python3 developmental_language_v15.py "fox grapes" --output report.json
python3 -m unittest -v test_developmental_language_v15.py
```

## v16

`experiments/lexical_research_v16.py` は、v15が自分で選んだ未知語と検索目的を実行します。English Wiktionary、Simple English Wiktionary、すでに読んだ児童文の用例を別の証拠として保持し、定義から複数のsense候補を抽出します。多義語は矛盾扱いせず代替senseとして残し、現在の用例に最も合うsenseを暫定採用します。

採用結果はv15の語彙記憶へ出典付きで書き戻されます。後の証拠で先頭senseが変われば訂正履歴を残し、学習済みの語を未知リストから外して次の不足語を選びます。

```bash
cd experiments
python3 lexical_research_v16.py "fox grapes" --output report.json
python3 -m unittest -v test_lexical_research_v16.py
```

## ライセンス

未設定です。利用・再配布条件を決めてからLICENSEを追加します。

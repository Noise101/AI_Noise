# AI_Noise

巨大LLMの縮小模倣ではなく、少ない経験から概念と因果関係を形成し、予測の失敗から自己訂正するAIアーキテクチャの実験プロジェクトです。

この方針は[ARCHITECTURE.md](ARCHITECTURE.md)の変更不能な設計契約として定義しています。任意のローカルAIは未検証候補の生成にしか使えず、証拠・採否・信念更新を担当できません。

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

通常開発では`python3 experiments/run_tests.py --profile quick --quiet`を使い、旧版を含む全評価は節目だけ実行します。ウェブ応答キャッシュ、要約出力、任意のローカルOllama補助を含む消費量方針は[RESOURCE_POLICY.md](RESOURCE_POLICY.md)を参照してください。

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
| v17 | 反復句の自律調査と構成的／非構成的意味の保留判定 |
| v18 | 日本語児童文からのtokenizerなし境界誘導と参照資料検証 |
| v19 | 日本語多義語の文脈接地・引用付き結論・反証訂正 |

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

## v17

`experiments/phrase_learning_v17.py` は、反復する2語・3語列から句候補を作り、単語とは別の未知意味として調査します。句ページが存在しなければ`unknown`として保存して次候補へ進みます。定義が見つかっても、構成語それぞれの意味が未接地なら熟語・イディオムとは断定しません。

実ウェブ例では`are sour`に独立した定義を発見できず、次の`at last`から`eventually_after_delay`を抽出しました。ただし定義資料がEnglish Wiktionary 1件だけだったため`single_source`であり、`at`と`last`の語義比較が済むまで非構成的とは認定しません。

```bash
cd experiments
python3 phrase_learning_v17.py "fox grapes" --max-phrases 4 --output report.json
python3 -m unittest -v test_phrase_learning_v17.py
```

## v18

`experiments/japanese_boundaries_v18.py` は、日本語Wikisourceを読み取り専用検索し、空白のない児童文から2〜6文字の反復列と左右文脈の多様性で語境界候補を作ります。日本語WiktionaryとWikipediaの完全ページ名／正式redirectで候補を検証し、部分一致検索による偽陽性は採用しません。

実ウェブ例では『イソップ童話集/きつねとつる』を自律選択し、`きつね`と`つる`だけが2資料一致の境界になりました。`ました`は1資料、`てしまいまし`などは0資料なので不採用です。`つる`はWikipedia上で曖昧語のため、語境界は確認できても「鶴／蔓」の意味は未確定として残します。

```bash
cd experiments
python3 japanese_boundaries_v18.py "きつね つる" --candidate-limit 15 --output report.json
python3 -m unittest -v test_japanese_boundaries_v18.py
```

## v19

`experiments/japanese_sense_grounding_v19.py` は、v18が見つけた意味曖昧な語について、日本語Wiktionaryの見出しから候補senseを動的に列挙し、Wikipediaの正式ページで参照証拠を補います。定義中の内容特徴と児童文の観察特徴を比較し、弱ければ全候補を残し、強ければ文脈限定で暫定採用します。後の反対特徴が優勢になれば訂正履歴を残します。

実ウェブ例では`つる`の候補として鶴・蔓・弦・鉉・動詞sense等を取得し、『きつねとつる』本文の`くちばし`反復から`鶴`を暫定採用しました。結論はWiktionaryとWikipediaを引用し、植物・茎・巻きつく等の新文脈で覆り得ることを保持します。

`--local-helper`を付けるとOllama 4Bへ未検証候補だけを依頼します。候補は証拠スコア0で、通常のsense台帳には入りません。実測では4Bが区別可能な候補を返せず0件となりましたが、通常経路はそのまま成功しました。

```bash
cd experiments
python3 japanese_sense_grounding_v19.py "きつね つる" --summary --output report.json
python3 -m unittest -v test_japanese_sense_grounding_v19.py
```

## v20

`experiments/autonomous_controller_v20.py` は、未解決の語・句・Why課題を期待情報利得で比較し、時間・手順数・実通信回数の上限内で次の調査を選びます。各周期の結果と採用した意味をJSONへ保存し、再起動時には完了IDだけでなく学習内容も語彙記憶へ復元します。

通信上限に達しても例外終了せず、`network_budget_exhausted`として状態を保存します。同じ資料は7日間のローカルキャッシュから読み、再実行時の通信とCodex出力を抑えます。

```bash
cd experiments
python3 autonomous_controller_v20.py "fox grapes" --state controller-state.json --max-steps 3 --max-network 8 --summary
```

同じコマンドを再実行すると、前回の知識から次の未解決課題へ進みます。これは常駐プロセスではなく、予算単位で安全に再開できる制御器です。

## v21: Codexを使わないローカル運転

日常の学習周期は`local_worker_v21.py`だけで実行できます。この経路はOpenAI APIやCodexを呼びません。通常のPython規則・証拠台帳・ウェブキャッシュで進み、状態、最新レポート、短いheartbeatを`.local/`へ保存します。現在のseedを学び切ると、読んだWikisourceページの未訪問リンクと証拠台帳の概念対から次のseed候補を生成し、選択理由と親URLを保存して自動遷移します。

```bash
cd experiments
python3 local_worker_v21.py start "fox grapes"
python3 local_worker_v21.py status
python3 local_worker_v21.py stop
```

`start`はバックグラウンドで一度だけ起動します。`status`の`heartbeat`が更新されていれば動作中です。現在のseed、処理済みカリキュラム数、`completed_gaps`、`remaining_gaps`、実通信数、停止理由を確認できます。通信上限は周期ごとのレート制限として自動再開されます。Codexは新しい学習機構の設計、失敗解析、節目のレビューだけに使う想定です。

## 漢字ペディア参照

`experiments/kanjipedia_reference_v22.py`は、漢字ペディアを日本語の漢字・熟語候補の完全一致検証に使います。保存するのは項目の存在、URL、応答ハッシュだけです。著作権のある辞書本文は知識へコピーせず、意味の判断にはWiktionary、Wikipedia、児童文の観察証拠も別々に必要です。サイト障害時は証拠なしとして通常経路を継続します。

## v23-v24: 知りたい欲と習得欲

`curiosity_drive_v23.py`は、未解決の文字・単語・熟語・会話行為・Why・概念について、遭遇回数、異なる題材での再出現、未解決期間、不確実性から「知りたい圧」を計算します。証拠不足なら欲は消えず、同じ証拠で同じ検索を反復せず、別文脈で再遭遇したときにさらに強くなります。会話は`said / asked / replied / answered / cried / quoth / told / saying`等を単語とは別の会話行為候補として観察します。

`mastery_drive_v24.py`は、Noise自身が文字、単語、熟語、会話、予測、因果、概念の能力を毎周期評価します。最弱層を次の習得目標として`mastery.json`とheartbeatへ出します。観察していない能力を満点にはせず、「完全な言語能力」ではなく、明示した証拠条件に対する現在の到達度だけを扱います。

## v25: ローカルAIとの会話練習

`local_conversation_v25.py`は、各新規カリキュラムで一度、Noise自身の最弱能力と最も強い好奇心から発話をテンプレート生成し、Ollama `qwen3:4b`と短い一往復を行います。返答、相手からの質問、観察した語形を`.local/dialogue-ledger.json`へ保存します。

ローカルAIは会話相手であり教師・採点者・情報源ではありません。全発言は`verified=false`、`evidence_score=0.0`で、辞書的意味や因果知識を直接更新できません。Ollamaが停止中なら会話だけを飛ばし、通常の学習は継続します。無効化する場合は`start`または`run`へ`--no-local-conversation`を付けます。

題材前線が空になると、過去に実際に読んだWikisource資料のURLから作品集接頭辞を発見し、その作品集の未訪問ページを次候補として補充します。これは固定の正解ページ一覧ではなく、観察済みの本棚から未知の本を選ぶ処理です。

読み取りタイムアウト、接続切断、HTTP 429/5xxは学習失敗にせず、状態を保存したまま最大30秒まで指数的に間隔を空けて自動再試行します。停止ファイルは待機中も確認します。プログラム欠陥や壊れた状態形式は再試行で隠さず`error`として停止します。

## ライセンス

未設定です。利用・再配布条件を決めてからLICENSEを追加します。

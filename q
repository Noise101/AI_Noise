#!/usr/bin/env bash
# Noise 日本語読書 — 保護者チェックの質問確認と回答
#
#   ./q              未回答の質問を表示
#   ./q 2 はい 1     位置対応で回答（1問目=2, 2問目=はい, 3問目=1）
#   ./q "2 はい 1"   （クオートしてもよい）
#
# 回答は数字（選択肢番号）か はい / いいえ。分からない問いは適当な語を
# 入れず飛ばす（そのまま未回答で残り、次のバッチで失効する）。

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/experiments"
runtime="$here/.local"

if [ "$#" -eq 0 ]; then
    out="$(python3 japanese_reader_v1.py status --runtime "$runtime")"
    if printf '%s\n' "$out" | grep -q '^── 質問'; then
        printf '%s\n' "$out" | sed -n '/^── 質問/,$p'
        echo
        echo 'この端末で答える:  ./q <答え1> <答え2> ...'
    else
        echo '未回答の質問はありません。'
        printf '%s\n' "$out" | grep -E '^(保護者確認|読解レベル|直近の読書)' || true
    fi
    exit 0
fi

python3 japanese_reader_v1.py answer "$*" --runtime "$runtime"

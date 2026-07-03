# 箏符コンバーター — デプロイ手順

## PythonAnywhere（完全無料）へのデプロイ

### 1. アカウント作成
https://www.pythonanywhere.com → 「Start running Python online in less than a minute!」
→ 「Create a Beginner account」（無料）でアカウント作成

### 2. ファイルをアップロード
「Files」タブ → 「Upload a file」で以下をアップロード：
- app.py
- requirements.txt
- templates/index.html（templatesフォルダを先に作成）

または「Bash Console」を開いて：
```bash
mkdir koto-flask
mkdir koto-flask/templates
cd koto-flask
```

### 3. ライブラリインストール
「Bash Console」で：
```bash
cd koto-flask
pip install --user flask pillow numpy opencv-python-headless
```

### 4. Webアプリ設定
「Web」タブ → 「Add a new web app」
→ Flask → Python 3.10
→ Source code: /home/あなたのID/koto-flask
→ WSGI file に以下を記入：
```python
import sys
sys.path.insert(0, '/home/あなたのID/koto-flask')
from app import app as application
```

### 5. 完成！
あなた専用のURLが発行されます：
https://あなたのID.pythonanywhere.com

---
## 機能
- 楽譜画像（JPG/PNG）をアップロード
- 調弦設定（C調/G調/平調子など）
- ClaudeのチャットからJSONをコピペ→自動変換
- 箏符漢字（一〜巾、△弱押し、▲強押し）を五線の外に表示
- 画像ダウンロード・印刷

## 技術
- Python + Flask + OpenCV + Pillow
- 五線自動検出（画像処理）
- ト音記号自動除外（縦線検出）
- Y座標数学的計算（G4=第2線基準）

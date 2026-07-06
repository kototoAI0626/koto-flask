"""
箏符コンバーター app.py
- 全ページ音符を先に分析 → 調弦パターンと壱の音を自動決定
- 画像拡大による安定した符頭検出
- 五線の実座標から正確な音高判定
- 和音の縦表示
- 調弦表をp1先頭に追加
"""
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────
CHROMATIC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ENHARMONIC = {'Bb':'A#','Eb':'D#','Ab':'G#','Db':'C#','Gb':'F#','B#':'C','E#':'F'}
STRING_KANJI = ['壱','弍','参','四','五','六','七','八','九','十','斗','為','巾']

# 各調子の音程パターン（壱からの半音インターバル）
TUNING_PATTERNS = {
    '平調子':    [0, 7, 9, 10, 14, 15, 19, 21, 22, 26, 27, 31, 33],
    '雲井調子':  [0, 7, 8, 12, 14, 15, 19, 20, 24, 26, 27, 31, 33],
    '本雲井調子':[0, 7, 8, 12, 14, 15, 19, 20, 24, 26, 27, 31, 32],
    '楽調子':    [0, 7, 9, 12, 14, 16, 19, 21, 24, 26, 28, 31, 33],
    '乃木調子':  [0, 7, 9, 11, 14, 16, 19, 21, 23, 26, 28, 31, 33],
    '中空調子':  [0, 7, 9, 10, 14, 16, 17, 21, 22, 26, 28, 29, 33],
    '古今調子':  [0, 7, 9, 12, 14, 15, 19, 21, 24, 26, 27, 31, 33],
}

# 壱の音の優先順位（G, A, D, F, C, Bb を優先）
ROOT_PRIORITY = {
    note: i for i, note in enumerate(['G','A','D','F','C','A#','E','B','F#','C#','G#','D#'])
}

FONT_PATHS = [
    '/usr/share/fonts/opentype/ipaexfont-mincho/ipaexm.ttf',
    '/usr/share/fonts/opentype/ipafont-mincho/ipamp.ttf',
    '/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf',
    '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf',
    '/usr/share/fonts/truetype/fonts-japanese-mincho.ttf',
    '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
]

# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
def note_to_midi(note, octave):
    n = ENHARMONIC.get(note, note)
    return (int(octave) + 1) * 12 + CHROMATIC.index(n)

def midi_to_note(midi):
    return CHROMATIC[midi % 12]

def get_font(size):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size, index=0)
            except:
                continue
    return ImageFont.load_default()

# ─────────────────────────────────────────────
# 調弦マップ構築
# ─────────────────────────────────────────────
def build_midi_map(pattern_name, root_midi, transpose=0):
    """調子名と壱のMIDI番号から調弦マップを構築"""
    pattern = TUNING_PATTERNS.get(pattern_name, TUNING_PATTERNS['楽調子'])
    m = {}
    for i, interval in enumerate(pattern):
        midi = root_midi + interval
        base = midi - transpose
        m[base]   = (STRING_KANJI[i], '')
        m[base+1] = (STRING_KANJI[i], '△')
        m[base+2] = (STRING_KANJI[i], '▲')
    return m

def get_tuning_display(pattern_name, root_midi):
    """チューナー用の調弦表示データを返す"""
    pattern = TUNING_PATTERNS.get(pattern_name, TUNING_PATTERNS['楽調子'])
    result = []
    for i, interval in enumerate(pattern):
        midi = root_midi + interval
        result.append({'string': STRING_KANJI[i], 'note': midi_to_note(midi)})
    return result

# 調号から調性の音階を定義
KEY_SIGNATURES_SCALE = {
    0:  ['C','D','E','F','G','A','B'],
    1:  ['G','A','B','C','D','E','F#'],
    2:  ['D','E','F#','G','A','B','C#'],
    3:  ['A','B','C#','D','E','F#','G#'],
    4:  ['E','F#','G#','A','B','C#','D#'],
    5:  ['B','C#','D#','E','F#','G#','A#'],
    -1: ['F','G','A','A#','C','D','E'],
    -2: ['A#','C','D','D#','F','G','A'],
    -3: ['D#','F','G','G#','A#','C','D'],
    -4: ['G#','A#','C','C#','D#','F','G'],
}
KEY_NAMES_MAP = {
    0: 'C長調', 1: 'G長調', 2: 'D長調', 3: 'A長調',
    4: 'E長調', 5: 'B長調',
    -1: 'F長調', -2: 'Bb長調', -3: 'Eb長調', -4: 'Ab長調',
}

def detect_key_signature(img_bytes_or_path):
    """
    楽譜画像から調号（♯・♭の数）を自動検出する。
    返り値: 整数（正=♯の数、負=♭の数、ゼロ=調号なし）
    """
    if isinstance(img_bytes_or_path, str):
        img = cv2.imread(img_bytes_or_path)
    else:
        nparr = np.frombuffer(img_bytes_or_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    hk_len = max(50, W // 8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hk_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    row_sums = np.sum(h_lines, axis=1) / 255
    threshold = W * 0.30
    line_rows = np.where(row_sums > threshold)[0]
    if len(line_rows) == 0:
        return 0
    groups = []
    cur = [line_rows[0]]
    for y in line_rows[1:]:
        if y - cur[-1] <= 3:
            cur.append(y)
        else:
            groups.append(int(np.mean(cur)))
            cur = [y]
    groups.append(int(np.mean(cur)))
    staves = []
    i = 0
    while i + 4 < len(groups):
        five = groups[i:i+5]
        gaps = [five[j+1]-five[j] for j in range(4)]
        avg_gap = np.mean(gaps)
        max_dev = max(abs(g - avg_gap) for g in gaps)
        if avg_gap > 3 and max_dev < avg_gap * 0.45:
            staves.append(five)
            i += 5
        else:
            i += 1
    treble_staves = staves[::2]
    if not treble_staves:
        return 0
    stave = treble_staves[0]
    gaps_list = [stave[j+1]-stave[j] for j in range(4)]
    avg_gap = float(np.mean(gaps_list))
    # ト音記号の右端を縦線検出で特定
    y1, y2 = stave[0]-2, stave[4]+2
    col_sums = np.sum(binary[y1:y2, :], axis=0) / 255
    vert_threshold = (y2-y1) * 0.6
    vert_cols = np.where(col_sums > vert_threshold)[0]
    if len(vert_cols) > 0:
        first_vert_groups = []
        cur_g = [vert_cols[0]]
        for xv in vert_cols[1:]:
            if xv - cur_g[-1] <= 5:
                cur_g.append(xv)
            else:
                first_vert_groups.append(cur_g)
                cur_g = [xv]
        first_vert_groups.append(cur_g)
        first_vert_end = int(np.mean(first_vert_groups[0])) + 5
    else:
        first_vert_end = int(W * 0.05)
    clef_width = int(avg_gap * 4.5)
    key_x_start = first_vert_end + clef_width
    key_x_end = int(W * 0.28)
    if key_x_end - key_x_start < avg_gap * 2:
        return 0
    y_top = max(0, stave[0] - int(avg_gap * 2))
    y_bot = min(H, stave[4] + int(avg_gap * 2))
    roi = gray[y_top:y_bot, key_x_start:key_x_end]
    roi_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    roi_h, roi_w = roi_bin.shape
    if roi_w < 10:
        return 0
    hk2 = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, roi_w//3), 1))
    staff_mask = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, hk2)
    roi_clean = cv2.subtract(roi_bin, staff_mask)
    contours, _ = cv2.findContours(roi_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sharp_count = 0
    flat_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 3:
            continue
        aspect = w / h if h > 0 else 0
        fill = area / (w * h) if w * h > 0 else 0
        if h > avg_gap * 3.5:
            continue  # 拍子記号を除外
        if (avg_gap * 1.5 <= h <= avg_gap * 3.0 and
            0.5 <= aspect <= 1.2 and
            0.15 <= fill <= 0.45):
            sharp_count += 1
        elif (avg_gap * 1.5 <= h <= avg_gap * 3.0 and
              0.3 <= aspect <= 0.7 and
              fill >= 0.30):
            flat_count += 1
    if sharp_count > flat_count:
        return min(sharp_count, 6)
    elif flat_count > sharp_count:
        return -min(flat_count, 4)
    else:
        return 0

def generate_scale_tuning(key_signature, min_midi=None):
    """
    調号から13弦の調弦を生成する。
    その調の音階（全音階）の音のみを使って弦を埋める。
    """
    scale = KEY_SIGNATURES_SCALE.get(key_signature, KEY_SIGNATURES_SCALE[0])
    if min_midi is None:
        min_midi = note_to_midi('E', 4)
    scale_notes = set(ENHARMONIC.get(n, n) for n in scale)
    # 壱の音を決定（最低使用音付近の音階の音）
    best_root = None
    for root in range(max(36, min_midi - 4), min(min_midi + 5, 68)):
        root_note = midi_to_note(root)
        if root_note in scale_notes:
            best_root = root
            break
    if best_root is None:
        best_root = min_midi
    # 13弦を音階の音で埋める
    strings = []
    current = best_root
    while len(strings) < 13 and current <= best_root + 36:
        note = midi_to_note(current)
        if note in scale_notes:
            strings.append(current)
        current += 1
    while len(strings) < 13:
        strings.append(strings[-1] + 12)
    tuning = [{'midi': m, 'note': midi_to_note(m), 'octave': m//12-1, 'string': STRING_KANJI[i]}
              for i, m in enumerate(strings[:13])]
    return tuning

def generate_free_tuning(used_midi_set):
    """
    楽譜の使用音から最適な13弦の配置を生成する（フリー調子）。

    各弦の開放弦をxとすると、x（開放弦）・x+1（弱押し）・x+2（強押し）の3音がカバーされる。
    13弦×3 = 最大39音をカバー可能。

    返り値: (tuning_list, unmatched_list, coverage_count)
      tuning_list: [{'midi': int, 'note': str, 'octave': int}, ...] 13弦分
      unmatched_list: カバーできなかったMIDIノートのリスト
      coverage_count: カバーできた音の数
    """
    if not used_midi_set:
        # デフォルト：楽調子・壱=G3
        root = note_to_midi('G', 3)
        pattern = TUNING_PATTERNS['楽調子']
        tuning = [{'midi': root+p, 'note': midi_to_note(root+p), 'octave': (root+p)//12-1} for p in pattern]
        return tuning, [], len(tuning)

    sorted_midi = sorted(used_midi_set)
    min_midi = min(sorted_midi)

    best_strings = None
    best_coverage = -1
    best_unmatched = list(sorted_midi)

    # 壱の音の候補（最低使用音の2半音下〜最低使用音まで）
    for root in range(max(36, min_midi - 2), min(min_midi + 1, 68)):
        strings = [root]
        covered = {root, root+1, root+2}
        remaining = [m for m in sorted_midi if m not in covered]

        while len(strings) < 13 and remaining:
            prev = strings[-1]
            target = remaining[0]

            best_gain = -1
            best_next = None

            for next_midi in range(prev + 1, min(prev + 8, target + 3)):
                new_covered = {next_midi, next_midi+1, next_midi+2}
                gain = len(new_covered & set(remaining))
                if gain > best_gain or (gain == best_gain and best_next is None):
                    best_gain = gain
                    best_next = next_midi

            if best_next is None:
                best_next = min(prev + 1, target)

            strings.append(best_next)
            new_covered = {best_next, best_next+1, best_next+2}
            covered |= new_covered
            remaining = [m for m in remaining if m not in covered]

        # 残りの弦を埋める
        while len(strings) < 13:
            strings.append(strings[-1] + 2)

        # カバー率を計算
        all_covered = set()
        for s in strings:
            all_covered |= {s, s+1, s+2}
        matched = used_midi_set & all_covered

        if len(matched) > best_coverage:
            best_coverage = len(matched)
            best_strings = strings[:13]
            best_unmatched = sorted(used_midi_set - all_covered)

    if best_strings is None:
        best_strings = list(range(min_midi, min_midi + 13))

    tuning = [{'midi': m, 'note': midi_to_note(m), 'octave': m//12-1} for m in best_strings]
    return tuning, best_unmatched, best_coverage

def build_midi_map_from_free_tuning(tuning, transpose=0):
    """フリー調子の調弦マップを構範する"""
    m = {}
    for i, t in enumerate(tuning):
        midi = t['midi']
        base = midi - transpose
        m[base]   = (STRING_KANJI[i], '')
        m[base+1] = (STRING_KANJI[i], '△')
        m[base+2] = (STRING_KANJI[i], '▲')
    return m

def suggest_tuning_smart(used_midi_set):
    """
    全ページの使用音から最適な（調子名, 壱のMIDI）を提案する。
    壱の音はC3〜G4の範囲でスライドして最適値を探す。
    """
    if not used_midi_set:
        return '楽調子', note_to_midi('G', 3), []

    best_name = '楽調子'
    best_root = note_to_midi('G', 3)
    best_score = -999999
    best_unmatched = list(used_midi_set)

    root_min = note_to_midi('C', 3)
    root_max = note_to_midi('G', 4)

    for pattern_name, pattern in TUNING_PATTERNS.items():
        for root_midi in range(root_min, root_max + 1):
            available = set()
            for interval in pattern:
                m = root_midi + interval
                available.add(m)
                available.add(m + 1)   # 弱押し
                available.add(m + 2)   # 強押し
                available.add(m - 1)   # 柱微調整

            matched = used_midi_set & available
            unmatched = used_midi_set - available

            # スコア：未対応音0が最優先、次にマッチ数最大、次に壱の音の優先度
            root_note = midi_to_note(root_midi)
            priority_bonus = -ROOT_PRIORITY.get(root_note, 12)
            score = len(matched) * 10 - len(unmatched) * 100 + priority_bonus

            if score > best_score:
                best_score = score
                best_name = pattern_name
                best_root = root_midi
                best_unmatched = sorted(unmatched)

    return best_name, best_root, best_unmatched

# ─────────────────────────────────────────────
# 五線譜処理
# ─────────────────────────────────────────────
def detect_staff_lines(gray):
    """適応的二値化 + 水平線検出で五線を検出する"""
    H, W = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    hk_len = max(50, W // 8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hk_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    row_sums = np.sum(h_lines, axis=1) / 255
    threshold = W * 0.30
    line_rows = np.where(row_sums > threshold)[0]
    if len(line_rows) == 0:
        return [], binary
    groups = []
    cur = [line_rows[0]]
    for y in line_rows[1:]:
        if y - cur[-1] <= 3:
            cur.append(y)
        else:
            groups.append(int(np.mean(cur)))
            cur = [y]
    groups.append(int(np.mean(cur)))
    staves = []
    i = 0
    while i + 4 < len(groups):
        five = groups[i:i+5]
        gaps = [five[j+1]-five[j] for j in range(4)]
        avg_gap = np.mean(gaps)
        max_dev = max(abs(g - avg_gap) for g in gaps)
        if avg_gap > 3 and max_dev < avg_gap * 0.45:
            staves.append(five)
            i += 5
        else:
            i += 1
    return staves[::2], binary  # ト音記号段のみ

def y_to_pitch(cy, stave):
    """
    符頭のY座標から音名を正確に判定する。

    各線・各間のY座標を直接計算し、最も近い位置の音名を返す。
    同距離の場合は五線上の音（線・間）を優先する。
    """
    g = [stave[j+1] - stave[j] for j in range(4)]

    # 五線上の音（優先度高）
    on_staff = [
        (float(stave[4]),             'F5'),
        ((stave[3]+stave[4]) / 2.0,   'E5'),
        (float(stave[3]),             'D5'),
        ((stave[2]+stave[3]) / 2.0,   'C5'),
        (float(stave[2]),             'B4'),
        ((stave[1]+stave[2]) / 2.0,   'A4'),
        (float(stave[1]),             'G4'),
        ((stave[0]+stave[1]) / 2.0,   'F4'),
        (float(stave[0]),             'E4'),
    ]

    # 加線領域の音（低優先度）
    ledger = [
        (stave[4] - g[3]*2.5, 'D6'),
        (stave[4] - g[3]*2,   'C6'),
        (stave[4] - g[3]*1.5, 'B5'),
        (stave[4] - g[3],     'A5'),
        (stave[4] - g[3]*0.5, 'G5'),
        (stave[0] + g[0]*0.5, 'D4'),
        (stave[0] + g[0],     'C4'),
        (stave[0] + g[0]*1.5, 'B3'),
        (stave[0] + g[0]*2,   'A3'),
        (stave[0] + g[0]*2.5, 'G3'),
    ]

    # 五線上の最近距離
    best_staff = min(on_staff, key=lambda p: abs(p[0]-cy))
    best_staff_dist = abs(best_staff[0] - cy)

    # 加線領域の最近距離
    best_ledger = min(ledger, key=lambda p: abs(p[0]-cy))
    best_ledger_dist = abs(best_ledger[0] - cy)

    # 五線上の音を優先するルール：
    # 1. 五線上の音が加線の音より近い場合 → 五線上を選ぶ
    # 2. 加線の音が明らかに近い場合（五線上より2px以上近い）→ 加線を選ぶ
    # 3. 同距離または五線上が少し遠い場合 → 五線上を優先
    if best_ledger_dist < best_staff_dist - 2.0:
        return best_ledger[1]
    else:
        return best_staff[1]

def prepare_image(img_bytes):
    """
    画像を読み込み、五線間隔が20pxになるよう拡大して返す。
    Returns: (orig_img, scaled_img, treble_staves, binary, scale, avg_gap)
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if orig_img is None:
        return None, None, None, None, 1.0, 10

    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    orig_H, orig_W = gray.shape

    # 五線検出して間隔を測定
    staves_orig, _ = detect_staff_lines(gray)
    if staves_orig:
        all_gaps = [staves_orig[s][j+1]-staves_orig[s][j]
                    for s in range(len(staves_orig)) for j in range(4)]
        avg_gap_orig = float(np.median(all_gaps))
    else:
        avg_gap_orig = max(8.0, orig_H / 60.0)

    # 目標五線間隔20pxになるよう拡大
    TARGET_GAP = 20.0
    scale = min(3.0, max(1.0, TARGET_GAP / avg_gap_orig))

    if scale > 1.05:
        new_W = int(orig_W * scale)
        new_H = int(orig_H * scale)
        scaled = cv2.resize(orig_img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
        gray_s = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    else:
        scale = 1.0
        scaled = orig_img
        gray_s = gray

    treble_staves, binary = detect_staff_lines(gray_s)
    if not treble_staves:
        return orig_img, scaled, [], binary, scale, avg_gap_orig

    all_gaps2 = [treble_staves[s][j+1]-treble_staves[s][j]
                 for s in range(len(treble_staves)) for j in range(4)]
    avg_gap = float(np.median(all_gaps2))

    return orig_img, scaled, treble_staves, binary, scale, avg_gap

def remove_non_noteheads(binary, treble_staves, avg_gap, W):
    """
    改良版五線除去。
    五線のY座標を正確に特定して1〜2px分だけ消す。
    符頭は五線より太いので形が残る。
    """
    result = binary.copy()
    H = result.shape[0]

    # 全五線のY座標を収集して消す（±1px）
    for stave in treble_staves:
        for y in stave:
            for dy in range(-1, 2):
                yy = y + dy
                if 0 <= yy < H:
                    result[yy, :] = 0

    # 符幹を除去（縦線、五線間隔の2倍以上）
    vk_len = max(int(avg_gap * 2.0), 20)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_len))
    result = cv2.subtract(result, cv2.morphologyEx(result, cv2.MORPH_OPEN, vk))

    # 梁を除去（五線間隔の0.6倍以上の横線）
    # 注意：大きすぎると符頭も削れるため縚めに設定
    bk_len = max(int(avg_gap * 0.6), 6)
    bk = cv2.getStructuringElement(cv2.MORPH_RECT, (bk_len, 1))
    result = cv2.subtract(result, cv2.morphologyEx(result, cv2.MORPH_OPEN, bk))

    # クロージングで符頭を補完
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, ck)

    return result

def is_rest(cnt, gap):
    """休符かどうかを判定する（符頭と誤検出しないため）"""
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    aspect = w / h if h > 0 else 0
    fill = area / (w * h) if w * h > 0 else 0
    # 八分休符：細い縦線
    if aspect < 0.35 and h > gap * 0.5:
        return True
    # 全音符・二分音符休符：横長の塗りつぶし長方形
    if aspect > 2.5 and fill > 0.65 and h < gap * 0.6:
        return True
    # 四分休符：複雑な形状（充填率が低い）
    if fill < 0.15 and area > gap * gap * 0.1:
        return True
    return False

def is_notehead(cnt, gap):
    """符頭かどうかを判定する（休符を除外）
    テスト済み最良設定：bk=0.6+fill=0.15+sol=0.40
    """
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area < 5: return False
    aspect = w / h if h > 0 else 0
    # 休符を除外
    if is_rest(cnt, gap): return False
    if not (gap * 0.45 <= w <= gap * 2.2): return False
    if not (gap * 0.35 <= h <= gap * 1.5): return False
    if not (0.55 <= aspect <= 2.2): return False
    if area < gap * gap * 0.12: return False  # 緩めた（0.15→ 0.12）
    fill = area / (w * h) if w * h > 0 else 0
    if fill < 0.15: return False  # 緩めた（0.22→0.15）
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and area / hull_area < 0.40: return False  # 緩めた（0.48→0.40）
    return True

def get_clef_end(stave, binary, W, is_first_page_first_stave=False):
    """ト音記号・拍子記号エリアの右端を検出"""
    clef_ratio = 0.20 if is_first_page_first_stave else 0.12
    y1, y2 = stave[0]-2, stave[4]+2
    cs = np.sum(binary[y1:y2, :], axis=0) / 255
    th = (y2-y1) * 0.65
    vc = np.where(cs > th)[0]
    valid = vc[vc > int(W * 0.05)]
    if len(valid) > 0:
        gps, cu = [], [valid[0]]
        for xv in valid[1:]:
            if xv - cu[-1] <= 10: cu.append(xv)
            else: gps.append(cu); cu = [xv]
        gps.append(cu)
        clef_end = max(int(np.mean(gps[0])) + 14, int(W * clef_ratio))
    else:
        clef_end = int(W * clef_ratio)
    return clef_end

def extract_pitches_from_page(img_bytes):
    """1ページから使用音MIDIセットを抽出する"""
    orig_img, scaled, treble_staves, binary, scale, avg_gap = prepare_image(img_bytes)
    if not treble_staves:
        return set()

    new_H, new_W = scaled.shape[:2]
    clean = remove_non_noteheads(binary, treble_staves, avg_gap, new_W)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    used_midi = set()
    for si, stave in enumerate(treble_staves):
        gap = np.mean([stave[j+1]-stave[j] for j in range(4)])
        y_top = max(0, stave[0] - int(gap * 3)) if si == 0 else (treble_staves[si-1][4] + stave[0]) // 2
        y_bot = min(new_H, stave[4] + int(gap * 3)) if si == len(treble_staves)-1 else (stave[4] + treble_staves[si+1][0]) // 2
        clef_end = get_clef_end(stave, binary, new_W, si == 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x+w//2, y+h//2
            if not (y_top <= cy <= y_bot): continue
            if cx < clef_end: continue
            if not is_notehead(cnt, gap): continue
            pitch = y_to_pitch(cy, stave)
            midi = note_to_midi(pitch[:-1], int(pitch[-1]))
            used_midi.add(midi)

    return used_midi

def group_chords(notes):
    """同じX座標付近の音符を和音としてグループ化する"""
    if not notes:
        return []
    threshold = 20
    groups = []
    current = [notes[0]]
    for n in notes[1:]:
        if abs(n['x'] - current[0]['x']) <= threshold and n['stave'] == current[0]['stave']:
            current.append(n)
        else:
            groups.append(sorted(current, key=lambda x: -x['y']))
            current = [n]
    groups.append(sorted(current, key=lambda x: -x['y']))
    return groups

def process_single_page(img_bytes, pattern_name, root_midi, transpose=0, font_size=20,
                        add_table_top=False, free_tuning=None):
    """
    1ページを処理して箏符付き画像を返す。
    free_tuning: フリー調子の場合は[{'midi':int,'note':str,...}]のリストを渡す。
    """
    orig_img, scaled, treble_staves, binary, scale, avg_gap = prepare_image(img_bytes)
    if not treble_staves:
        return None, 0

    new_H, new_W = scaled.shape[:2]
    clean = remove_non_noteheads(binary, treble_staves, avg_gap, new_W)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # フリー調子または既存調子パターンから調弦マップを構範
    if free_tuning:
        midi_map = build_midi_map_from_free_tuning(free_tuning, transpose)
    else:
        midi_map = build_midi_map(pattern_name, root_midi, transpose)
    font = get_font(font_size)

    notes = []
    for si, stave in enumerate(treble_staves):
        gap = np.mean([stave[j+1]-stave[j] for j in range(4)])
        y_top = max(0, stave[0] - int(gap * 3)) if si == 0 else (treble_staves[si-1][4] + stave[0]) // 2
        y_bot = min(new_H, stave[4] + int(gap * 3)) if si == len(treble_staves)-1 else (stave[4] + treble_staves[si+1][0]) // 2
        clef_end = get_clef_end(stave, binary, new_W, si == 0 and add_table_top)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x+w//2, y+h//2
            if not (y_top <= cy <= y_bot): continue
            if cx < clef_end: continue
            if not is_notehead(cnt, gap): continue
            pitch = y_to_pitch(cy, stave)
            midi = note_to_midi(pitch[:-1], int(pitch[-1]))
            if midi in midi_map:
                kanji, suffix = midi_map[midi]
                notes.append({'x': int(cx/scale), 'y': int(cy/scale),
                               'stave': si, 'kanji': kanji, 'suffix': suffix,
                               'stave_orig': [int(s/scale) for s in stave]})

    notes.sort(key=lambda n: (n['stave'], n['x']))
    chord_groups = group_chords(notes)

    img_pil = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    drawn = 0
    for group in chord_groups:
        stave_orig = group[0]['stave_orig']
        top_y = stave_orig[0] - 8
        cx = group[0]['x']

        if len(group) == 1:
            n = group[0]
            full = n['kanji'] + n['suffix']
            col = (180,0,0) if not n['suffix'] else (26,106,170) if n['suffix']=='△' else (122,26,170)
            bbox = draw.textbbox((0,0), full, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            tx = cx - tw//2
            ty = top_y - th
            draw.rectangle([tx-2,ty-2,tx+tw+2,ty+th+2], fill=(255,255,255,210))
            draw.text((tx,ty), full, font=font, fill=col)
            drawn += 1
        else:
            texts = []
            for n in group:
                full = n['kanji'] + n['suffix']
                col = (180,0,0) if not n['suffix'] else (26,106,170) if n['suffix']=='△' else (122,26,170)
                bbox = draw.textbbox((0,0), full, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                texts.append({'full':full,'col':col,'tw':tw,'th':th})
            max_tw = max(t['tw'] for t in texts)
            total_th = sum(t['th'] for t in texts) + 2*(len(texts)-1)
            tx = cx - max_tw//2
            ty = top_y - total_th
            draw.rectangle([tx-2,ty-2,tx+max_tw+2,top_y+2], fill=(255,255,255,210))
            cur_y = ty
            for t in texts:
                draw.text((cx-t['tw']//2, cur_y), t['full'], font=font, fill=t['col'])
                cur_y += t['th'] + 2
            drawn += len(group)

    result_img = img_pil

    if add_table_top:
        # フリー調子または既存調子の調弦表を追加
        if free_tuning:
            tuning_display_for_table = [{'string': t.get('string', STRING_KANJI[i]), 'note': t['note']}
                                        for i, t in enumerate(free_tuning)]
            result_img = add_tuning_table_top_from_display(
                result_img, 'カスタム調弦', tuning_display_for_table, font_size
            )
        else:
            result_img = add_tuning_table_top(result_img, pattern_name, root_midi, font_size)

    return result_img, drawn

def add_tuning_table_top(img, pattern_name, root_midi, font_size):
    """画像の上部に調弦表を追加する（既存調子用）"""
    tuning_display = get_tuning_display(pattern_name, root_midi)
    root_note = midi_to_note(root_midi)
    title = f'調弦表 — {pattern_name}（壱={root_note}）'
    return add_tuning_table_top_from_display(img, title, tuning_display, font_size)

def add_tuning_table_top_from_display(img, title, tuning_display, font_size):
    """画像の上部に調弦表を追加する（汎用）"""
    W, H = img.size
    table_font_size = max(11, min(font_size, 16))
    tf = get_font(table_font_size)
    tf_title = get_font(table_font_size + 1)
    padding = 6
    cell_w = max(34, W // 14)
    cell_h = table_font_size * 2 + padding * 3
    title_h = table_font_size + padding * 2
    table_h = title_h + cell_h + padding
    table_img = Image.new('RGB', (W, table_h), (248, 244, 236))
    td = ImageDraw.Draw(table_img)
    td.line([(0, table_h-2), (W, table_h-2)], fill=(180, 150, 100), width=2)
    td.text((padding, padding), title, font=tf_title, fill=(100, 50, 0))
    y0 = title_h
    for i, d in enumerate(tuning_display):
        x = i * cell_w
        bg = (255, 252, 242) if i % 2 == 0 else (242, 248, 255)
        td.rectangle([x+1, y0+1, x+cell_w-1, y0+cell_h-1], fill=bg, outline=(190, 165, 120))
        bbox = td.textbbox((0,0), d['string'], font=tf)
        kw = bbox[2]-bbox[0]
        td.text((x+(cell_w-kw)//2, y0+padding), d['string'], font=tf, fill=(160, 30, 0))
        bbox2 = td.textbbox((0,0), d['note'], font=tf)
        nw = bbox2[2]-bbox2[0]
        td.text((x+(cell_w-nw)//2, y0+padding+table_font_size+2), d['note'], font=tf, fill=(20, 60, 140))
    combined = Image.new('RGB', (W, H + table_h), (255, 255, 255))
    combined.paste(table_img, (0, 0))
    combined.paste(img, (0, table_h))
    return combined

# ─────────────────────────────────────────────
# Flask ルート
# ─────────────────────────────────────────────
@app.route('/')
def index():
    pattern_names = list(TUNING_PATTERNS.keys())
    return render_template('index.html', patterns=pattern_names)

@app.route('/analyze_multi', methods=['POST'])
def analyze_multi():
    """全ページを分析して最適調弦を提案する。
    free_tuning=trueの場合はフリー調子を生成する。
    """
    try:
        files = request.files.getlist('images')
        use_free = request.form.get('free_tuning', 'false') == 'true'
        if not files:
            return jsonify({'error': '画像が必要です'}), 400

        all_used_midi = set()
        page_count = 0
        all_img_bytes = []  # 調号検出用に保存
        for f in files:
            img_bytes = f.read()
            if not img_bytes: continue
            all_img_bytes.append(img_bytes)
            used = extract_pitches_from_page(img_bytes)
            all_used_midi |= used
            page_count += 1

        if not all_used_midi:
            return jsonify({'error': '五線譜が検出できませんでした'}), 400

        used_names = sorted(set(midi_to_note(m) for m in all_used_midi))

        if use_free:
            # 調号検出ベースの調弦生成
            # 1ページ目の画像から調号を検出
            key_sig = 0
            if all_img_bytes:
                key_sig = detect_key_signature(all_img_bytes[0])

            # 使用音の最低音を基準に壱の音を決定
            min_midi = min(all_used_midi) if all_used_midi else note_to_midi('E', 4)
            free_tuning = generate_scale_tuning(key_sig, min_midi)

            # カバー率を計算
            all_covered = set()
            for t in free_tuning:
                m = t['midi']
                all_covered |= {m, m+1, m+2}
            coverage = len(all_used_midi & all_covered)
            unmatched = sorted(all_used_midi - all_covered)

            for i, t in enumerate(free_tuning):
                t['string'] = STRING_KANJI[i]
            tuning_display = [{'string': t['string'], 'note': t['note']} for t in free_tuning]
            unmatched_names = sorted(set(midi_to_note(m) for m in unmatched))
            key_name = KEY_NAMES_MAP.get(key_sig, f'調号{key_sig:+d}')
            return jsonify({
                'mode': 'free',
                'free_tuning': free_tuning,
                'tuning_display': tuning_display,
                'used_notes': used_names,
                'unmatched_notes': unmatched_names,
                'coverage': coverage,
                'total_used': len(all_used_midi),
                'page_count': page_count,
                'key_name': key_name,
                'key_sig': key_sig,
            })
        else:
            # 既存調子パターンから提案
            pattern_name, root_midi, unmatched = suggest_tuning_smart(all_used_midi)
            tuning_display = get_tuning_display(pattern_name, root_midi)
            root_note = midi_to_note(root_midi)
            unmatched_names = sorted(set(midi_to_note(m) for m in unmatched))
            return jsonify({
                'mode': 'preset',
                'pattern_name': pattern_name,
                'root_note': root_note,
                'root_midi': root_midi,
                'tuning_display': tuning_display,
                'used_notes': used_names,
                'unmatched_notes': unmatched_names,
                'page_count': page_count,
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_multi', methods=['POST'])
def process_multi():
    """全ページに箏符を付与してページ別に返す。
    free_tuning_jsonがある場合はフリー調子を使用する。
    """
    try:
        import json as json_module
        files = request.files.getlist('images')
        pattern_name = request.form.get('pattern_name', '楽調子')
        root_midi = int(request.form.get('root_midi', note_to_midi('G', 3)))
        font_size = int(request.form.get('font_size', 20))
        transpose = int(request.form.get('transpose', 0))
        free_tuning_json = request.form.get('free_tuning_json', '')

        # フリー調子の場合
        free_tuning = None
        if free_tuning_json:
            try:
                free_tuning = json_module.loads(free_tuning_json)
            except:
                free_tuning = None

        if not files:
            return jsonify({'error': '画像が必要です'}), 400

        results = []
        total_notes = 0
        for i, f in enumerate(files):
            img_bytes = f.read()
            if not img_bytes: continue
            add_table_top = (i == 0)
            result_img, note_count = process_single_page(
                img_bytes, pattern_name, root_midi, transpose, font_size, add_table_top,
                free_tuning=free_tuning
            )
            if result_img is None:
                results.append(None)
                continue
            buf = io.BytesIO()
            result_img.save(buf, format='JPEG', quality=92)
            buf.seek(0)
            results.append(buf.read())
            total_notes += note_count

        tr_str = f'（移調{transpose:+d}半音）' if transpose != 0 else ''

        # 調弦表示用データを準備
        if free_tuning:
            for i, t in enumerate(free_tuning):
                if 'string' not in t:
                    t['string'] = STRING_KANJI[i]
            tuning_disp = [{'string': t['string'], 'note': t['note']} for t in free_tuning]
            pattern_label = 'カスタム調弦'
            root_note = free_tuning[0]['note'] if free_tuning else '?'
        else:
            tuning_disp = get_tuning_display(pattern_name, root_midi)
            pattern_label = pattern_name
            root_note = midi_to_note(root_midi)

        if len([r for r in results if r is not None]) == 1:
            img_data = next(r for r in results if r is not None)
            return jsonify({
                'success': True,
                'mode': 'single',
                'image': f'data:image/jpeg;base64,{base64.b64encode(img_data).decode()}',
                'message': f'{total_notes}個の音符を検出して箏符を付与しました{tr_str}',
                'tuning_display': tuning_disp,
                'pattern_name': pattern_label,
                'root_note': root_note,
            })

        images_b64 = []
        for i, img_data in enumerate(results):
            if img_data is not None:
                images_b64.append({
                    'page': i + 1,
                    'image': f'data:image/jpeg;base64,{base64.b64encode(img_data).decode()}',
                    'filename': f'箏符楽譜_p{i+1:02d}.jpg',
                })

        return jsonify({
            'success': True,
            'mode': 'multi',
            'images': images_b64,
            'page_count': len(images_b64),
            'message': f'{len(files)}ページ・{total_notes}個の音符を検出して箏符を付与しました{tr_str}',
            'tuning_display': tuning_disp,
            'pattern_name': pattern_label,
            'root_note': root_note,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tuning_info', methods=['GET'])
def tuning_info():
    pattern_name = request.args.get('pattern_name', '楽調子')
    root_midi = int(request.args.get('root_midi', note_to_midi('G', 3)))
    return jsonify({
        'pattern_name': pattern_name,
        'root_note': midi_to_note(root_midi),
        'tuning_display': get_tuning_display(pattern_name, root_midi),
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

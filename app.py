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

def build_pitch_y_map(stave):
    """五線の実座標から各音名のY座標マップを構築する"""
    g01 = stave[1] - stave[0]
    g12 = stave[2] - stave[1]
    g23 = stave[3] - stave[2]
    g34 = stave[4] - stave[3]
    return {
        'G3': stave[0] + g01 * 2.5,
        'A3': stave[0] + g01 * 2,
        'B3': stave[0] + g01 * 1.5,
        'C4': stave[0] + g01,
        'D4': stave[0] + g01 * 0.5,
        'E4': float(stave[0]),
        'F4': stave[0] - g01 * 0.5,
        'G4': float(stave[1]),
        'A4': stave[1] - g12 * 0.5,
        'B4': float(stave[2]),
        'C5': stave[2] - g23 * 0.5,
        'D5': float(stave[3]),
        'E5': stave[3] - g34 * 0.5,
        'F5': float(stave[4]),
        'G5': stave[4] - g34 * 0.5,
        'A5': stave[4] - g34,
        'B5': stave[4] - g34 * 1.5,
        'C6': stave[4] - g34 * 2,
        'D6': stave[4] - g34 * 2.5,
    }

def y_to_pitch(cy, stave):
    """Y座標から最も近い音名を返す"""
    pitch_y = build_pitch_y_map(stave)
    return min(pitch_y.keys(), key=lambda p: abs(pitch_y[p] - cy))

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

    # 五線除去後の残骸（短い水平線）を追加除去
    # 五線間隔の0.3倍以上の水平線を消す（符頭より短い横線を除去）
    stub_len = max(int(avg_gap * 0.3), 4)
    stub_k = cv2.getStructuringElement(cv2.MORPH_RECT, (stub_len, 1))
    stub_mask = cv2.morphologyEx(result, cv2.MORPH_OPEN, stub_k)
    result = cv2.subtract(result, stub_mask)

    # 符幹を除去（縦線、五線間隔の2倍以上）
    vk_len = max(int(avg_gap * 2.0), 20)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_len))
    result = cv2.subtract(result, cv2.morphologyEx(result, cv2.MORPH_OPEN, vk))

    # 梁を除去（五線間隔の1.0倍以上の横線）
    bk_len = max(int(avg_gap * 1.0), 10)
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
    """符頭かどうかを判定する（休符を除外）"""
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area < 5: return False
    aspect = w / h if h > 0 else 0
    # 休符を除外
    if is_rest(cnt, gap): return False
    if not (gap * 0.45 <= w <= gap * 2.2): return False
    if not (gap * 0.35 <= h <= gap * 1.5): return False
    if not (0.55 <= aspect <= 2.2): return False
    if area < gap * gap * 0.15: return False
    fill = area / (w * h) if w * h > 0 else 0
    if fill < 0.22: return False
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and area / hull_area < 0.48: return False
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
                        add_table_top=False):
    """1ページを処理して箏符付き画像を返す"""
    orig_img, scaled, treble_staves, binary, scale, avg_gap = prepare_image(img_bytes)
    if not treble_staves:
        return None, 0

    new_H, new_W = scaled.shape[:2]
    clean = remove_non_noteheads(binary, treble_staves, avg_gap, new_W)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        result_img = add_tuning_table_top(result_img, pattern_name, root_midi, font_size)

    return result_img, drawn

def add_tuning_table_top(img, pattern_name, root_midi, font_size):
    """画像の上部に調弦表を追加する（p1用）"""
    tuning_display = get_tuning_display(pattern_name, root_midi)
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
    root_note = midi_to_note(root_midi)
    title = f'調弦表 — {pattern_name}（壱={root_note}）'
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
    """全ページを分析して最適調弦を提案する"""
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': '画像が必要です'}), 400

        all_used_midi = set()
        page_count = 0
        for f in files:
            img_bytes = f.read()
            if not img_bytes: continue
            used = extract_pitches_from_page(img_bytes)
            all_used_midi |= used
            page_count += 1

        if not all_used_midi:
            return jsonify({'error': '五線譜が検出できませんでした'}), 400

        pattern_name, root_midi, unmatched = suggest_tuning_smart(all_used_midi)
        tuning_display = get_tuning_display(pattern_name, root_midi)
        root_note = midi_to_note(root_midi)
        used_names = sorted(set(midi_to_note(m) for m in all_used_midi))
        unmatched_names = sorted(set(midi_to_note(m) for m in unmatched))

        return jsonify({
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
    """全ページに箏符を付与してページ別に返す"""
    try:
        files = request.files.getlist('images')
        pattern_name = request.form.get('pattern_name', '楽調子')
        root_midi = int(request.form.get('root_midi', note_to_midi('G', 3)))
        font_size = int(request.form.get('font_size', 20))
        transpose = int(request.form.get('transpose', 0))

        if not files:
            return jsonify({'error': '画像が必要です'}), 400

        results = []
        total_notes = 0
        for i, f in enumerate(files):
            img_bytes = f.read()
            if not img_bytes: continue
            add_table_top = (i == 0)
            result_img, note_count = process_single_page(
                img_bytes, pattern_name, root_midi, transpose, font_size, add_table_top
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
        root_note = midi_to_note(root_midi)

        if len([r for r in results if r is not None]) == 1:
            img_data = next(r for r in results if r is not None)
            return jsonify({
                'success': True,
                'mode': 'single',
                'image': f'data:image/jpeg;base64,{base64.b64encode(img_data).decode()}',
                'message': f'{total_notes}個の音符を検出して箏符を付与しました{tr_str}',
                'tuning_display': get_tuning_display(pattern_name, root_midi),
                'pattern_name': pattern_name,
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
            'tuning_display': get_tuning_display(pattern_name, root_midi),
            'pattern_name': pattern_name,
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

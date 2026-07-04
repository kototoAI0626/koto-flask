from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 弦名（山田流・正式字体）
STRING_KANJI = ['壱','弍','参','四','五','六','七','八','九','十','斗','為','巾']

# 半音階
CHROMATIC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ENHARMONIC = {'Bb':'A#','Eb':'D#','Ab':'G#','Db':'C#','Gb':'F#','B#':'C','E#':'F'}

# ─────────────────────────────────────────────
# 箏の調弦定義（正確な音列・オクターブ付き）
# ─────────────────────────────────────────────
TUNING_DEFS = {
    '平調子':    [('D',4),('G',3),('A',3),('A#',3),('D',4),('D#',4),('G',4),('A',4),('A#',4),('D',5),('D#',5),('G',5),('A',5)],
    '雲井調子':  [('D',4),('G',3),('G#',3),('C',4), ('D',4),('D#',4),('G',4),('G#',4),('C',5), ('D',5),('D#',5),('G',5),('A',5)],
    '本雲井調子':[('D',4),('G',3),('G#',3),('C',4), ('D',4),('D#',4),('G',4),('G#',4),('C',5), ('D',5),('D#',5),('G',5),('G#',5)],
    '楽調子':    [('D',4),('G',3),('A',3),('C',4), ('D',4),('E',4), ('G',4),('A',4),('C',5), ('D',5),('E',5), ('G',5),('A',5)],
    '乃木調子':  [('D',4),('G',3),('A',3),('B',3), ('D',4),('E',4), ('G',4),('A',4),('B',4), ('D',5),('E',5), ('G',5),('A',5)],
    '中空調子':  [('D',4),('G',3),('A',3),('A#',3),('D',4),('E',4), ('F',4),('A',4),('A#',4),('D',5),('E',5), ('F',5),('A',5)],
    '古今調子':  [('D',4),('G',4),('A',3),('C',4), ('D',4),('D#',4),('G',4),('A',4),('C',5), ('D',5),('D#',5),('G',5),('A',5)],
    'G調':  [('G',3),('A',3),('B',3), ('C',4),('D',4),('E',4),('F',4),('G',4),('A',4),('B',4), ('C',5),('D',5),('E',5)],
    'D調':  [('D',4),('E',4),('F#',4),('G',4),('A',4),('B',4),('C#',5),('D',5),('E',5),('F#',5),('G',5),('A',5),('B',5)],
    'A調':  [('A',3),('B',3),('C#',4),('D',4),('E',4),('F#',4),('G#',4),('A',4),('B',4),('C#',5),('D',5),('E',5),('F#',5)],
    'C調':  [('C',4),('D',4),('E',4), ('F',4),('G',4),('A',4),('B',4), ('C',5),('D',5),('E',5), ('F',5),('G',5),('A',5)],
    'F調':  [('F',3),('G',3),('A',3), ('A#',3),('C',4),('D',4),('E',4),('F',4),('G',4),('A',4), ('A#',4),('C',5),('D',5)],
    'Bb調': [('A#',3),('C',4),('D',4),('D#',4),('F',4),('G',4),('A',4),('A#',4),('C',5),('D',5),('D#',5),('F',5),('G',5)],
}

PRESET_ORDER = ['G調','D調','A調','C調','F調','Bb調','平調子','雲井調子','本雲井調子','楽調子','乃木調子','中空調子','古今調子']

# ─────────────────────────────────────────────
# 五線譜上の音符位置→音名マッピング（ト音記号基準: 第2線=G4）
# ─────────────────────────────────────────────
PITCH_STEPS = {
    'B2':-7.5,'C3':-7,'D3':-6.5,'E3':-6,'F3':-5.5,'F#3':-5.5,
    'G3':-5,  'G#3':-5,'A3':-4.5,'A#3':-4.5,'B3':-4,
    'C4':-3.5,'C#4':-3.5,'D4':-3,'D#4':-3,'E4':-2.5,
    'F4':-2,  'F#4':-2,
    'G4':-1.5,'G#4':-1.5,'A4':-1,'A#4':-1,'B4':-0.5,
    'C5':0,   'C#5':0,'D5':0.5,'D#5':0.5,'E5':1,
    'F5':1.5, 'F#5':1.5,'G5':2,'G#5':2,'A5':2.5,'A#5':2.5,'B5':3,
    'C6':3.5, 'D6':4,
}
STEP_TO_PITCH = {}
for pitch, step in PITCH_STEPS.items():
    if step not in STEP_TO_PITCH:
        STEP_TO_PITCH[step] = pitch

# フォントパス（PythonAnywhere対応）
FONT_PATHS = [
    '/usr/share/fonts/opentype/ipaexfont-mincho/ipaexm.ttf',
    '/usr/share/fonts/opentype/ipafont-mincho/ipamp.ttf',
    '/usr/share/fonts/opentype/ipafont-mincho/ipam.ttf',
    '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf',
    '/usr/share/fonts/truetype/fonts-japanese-mincho.ttf',
    '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    '/usr/share/fonts/truetype/noto/NotoSerifCJK-Bold.ttc',
]

def get_font(size):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size, index=0)
            except:
                continue
    return ImageFont.load_default()

def note_to_midi(note, octave):
    n = ENHARMONIC.get(note, note)
    return (int(octave) + 1) * 12 + CHROMATIC.index(n)

def midi_to_name(midi):
    return CHROMATIC[midi % 12]

def build_midi_map(tuning_name, transpose=0):
    tuning = TUNING_DEFS.get(tuning_name, TUNING_DEFS['G調'])
    m = {}
    for i, (note, octave) in enumerate(tuning):
        midi = note_to_midi(note, octave)
        base = midi - transpose
        m[base]   = (STRING_KANJI[i], '')
        m[base+1] = (STRING_KANJI[i], '△')
        m[base+2] = (STRING_KANJI[i], '▲')
    return m

def get_tuning_display(tuning_name):
    tuning = TUNING_DEFS.get(tuning_name, TUNING_DEFS['G調'])
    result = []
    for i, (note, octave) in enumerate(tuning):
        n = ENHARMONIC.get(note, note)
        result.append({'string': STRING_KANJI[i], 'note': n})
    return result

# ─────────────────────────────────────────────
# 五線譜処理
# ─────────────────────────────────────────────
def detect_staves(binary, W):
    row_sums = np.sum(binary, axis=1) / 255
    cands = np.where(row_sums > W * 0.35)[0]
    if len(cands) == 0: return []
    grps, cur = [], [cands[0]]
    for y in cands[1:]:
        if y - cur[-1] <= 4: cur.append(y)
        else: grps.append(cur); cur = [y]
    grps.append(cur)
    lc = [int(np.mean(g)) for g in grps]
    staves = []
    i = 0
    while i + 4 < len(lc):
        five = lc[i:i+5]
        gaps = [five[j+1]-five[j] for j in range(4)]
        if np.mean(gaps) > 3 and max(gaps)-min(gaps) < np.mean(gaps)*0.6:
            staves.append(five); i += 5
        else: i += 1
    return staves[::2]  # ト音記号段のみ

def find_clef_end(stave, binary, W):
    """
    ト音記号・調号・拍子記号エリアの右端を検出。
    4/4などの拍子記号を除外するため最低20%を確保。
    """
    y1, y2 = stave[0]-2, stave[4]+2
    cs = np.sum(binary[y1:y2, :], axis=0) / 255
    th = (y2-y1) * 0.65
    vc = np.where(cs > th)[0]
    valid = vc[vc > int(W*0.06)]
    if len(valid) == 0:
        return int(W * 0.22)
    gps, cu = [], [valid[0]]
    for x in valid[1:]:
        if x - cu[-1] <= 8: cu.append(x)
        else: gps.append(cu); cu = [x]
    gps.append(cu)
    clef_end = int(np.mean(gps[0])) + 16
    # 最低20%を保証（拍子記号を確実に除外）
    return max(clef_end, int(W * 0.20))

def y_to_pitch(cy, stave):
    gap = np.mean([stave[j+1]-stave[j] for j in range(4)])
    g4_y = stave[1]
    step = (g4_y - cy) / gap
    steps = list(STEP_TO_PITCH.keys())
    nearest = min(steps, key=lambda s: abs(s - step))
    return STEP_TO_PITCH[nearest]

def is_notehead(cnt, gap):
    """
    符頭（音符の玉）かどうかを判定。
    検出漏れを減らすため許容範囲を緩めに設定。
    拍子記号の除外はfind_clef_endで行う。
    """
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    aspect = w/h if h > 0 else 0
    # サイズフィルタ（緩め）
    if not (gap*0.4 <= w <= gap*2.5): return False
    if not (gap*0.3 <= h <= gap*1.6): return False
    # アスペクト比
    if not (0.5 <= aspect <= 2.8): return False
    # 面積
    if area < gap*gap*0.15: return False
    # 充填率（緩め）
    fill_ratio = area / (w * h) if w*h > 0 else 0
    if fill_ratio < 0.28: return False
    # 凸性（緩め）
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        if area / hull_area < 0.45: return False
    return True

def add_tuning_table(img, tuning_name, font_size):
    """出力画像の下部に調弦表を追加する"""
    tuning = TUNING_DEFS.get(tuning_name, TUNING_DEFS['G調'])
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

    # 区切り線
    td.line([(0, 0), (W, 0)], fill=(180, 150, 100), width=2)

    # タイトル
    title = f'調弦表 — {tuning_name}'
    td.text((padding, padding), title, font=tf_title, fill=(100, 50, 0))

    # 各弦
    y0 = title_h
    for i, (note, octave) in enumerate(tuning):
        n = ENHARMONIC.get(note, note)
        kanji = STRING_KANJI[i]
        x = i * cell_w
        bg = (255, 252, 242) if i % 2 == 0 else (242, 248, 255)
        td.rectangle([x+1, y0+1, x+cell_w-1, y0+cell_h-1], fill=bg, outline=(190, 165, 120))
        # 弦名（漢字・赤系）
        bbox = td.textbbox((0,0), kanji, font=tf)
        kw = bbox[2]-bbox[0]
        td.text((x+(cell_w-kw)//2, y0+padding), kanji, font=tf, fill=(160, 30, 0))
        # 音名（英語・青系）
        bbox2 = td.textbbox((0,0), n, font=tf)
        nw = bbox2[2]-bbox2[0]
        td.text((x+(cell_w-nw)//2, y0+padding+table_font_size+2), n, font=tf, fill=(20, 60, 140))

    combined = Image.new('RGB', (W, H + table_h), (255, 255, 255))
    combined.paste(img, (0, 0))
    combined.paste(table_img, (0, H))
    return combined

# ─────────────────────────────────────────────
# 楽譜分析
# ─────────────────────────────────────────────
def extract_used_pitches(contours, treble_staves, binary, W):
    used_midi = set()
    for si, stave in enumerate(treble_staves):
        clef_end = find_clef_end(stave, binary, W)
        gap = np.mean([stave[j+1]-stave[j] for j in range(4)])
        H_total = binary.shape[0]
        y_top = 0 if si == 0 else (treble_staves[si-1][4]+stave[0])//2
        y_bot = H_total if si == len(treble_staves)-1 else (stave[4]+treble_staves[si+1][0])//2
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

def suggest_tuning(used_midi):
    if not used_midi:
        return 'G調', []
    best_name = None
    best_score = -1
    best_unmatched = []
    for name in PRESET_ORDER:
        tuning = TUNING_DEFS[name]
        available = set()
        for note, octave in tuning:
            m = note_to_midi(note, octave)
            available.add(m)
            available.add(m+1)
            available.add(m+2)
        matched = used_midi & available
        unmatched = used_midi - available
        score = len(matched) - len(unmatched) * 2
        if score > best_score:
            best_score = score
            best_name = name
            best_unmatched = sorted(unmatched)
    return best_name, best_unmatched

# ─────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────
def process_image(img_bytes, tuning_name, transpose=0, font_size=20, auto_suggest=False):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    H, W = img_cv.shape

    _, binary = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY_INV)
    treble_staves = detect_staves(binary, W)
    if not treble_staves:
        return None, "五線譜が検出できませんでした", None, []

    # 五線・符幹・梁を除去
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    hl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    bn = cv2.subtract(binary, hl)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vl = cv2.morphologyEx(bn, cv2.MORPH_OPEN, vk)
    bn = cv2.subtract(bn, vl)
    bk = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    bl = cv2.morphologyEx(bn, cv2.MORPH_OPEN, bk)
    bn_clean = cv2.subtract(bn, bl)

    contours, _ = cv2.findContours(bn_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suggested_tuning = None
    if auto_suggest:
        used_midi = extract_used_pitches(contours, treble_staves, binary, W)
        suggested_tuning, _ = suggest_tuning(used_midi)
        tuning_name = suggested_tuning

    midi_map = build_midi_map(tuning_name, transpose)
    font = get_font(font_size)

    notes = []
    for si, stave in enumerate(treble_staves):
        clef_end = find_clef_end(stave, binary, W)
        gap = np.mean([stave[j+1]-stave[j] for j in range(4)])
        y_top = 0 if si == 0 else (treble_staves[si-1][4]+stave[0])//2
        y_bot = H if si == len(treble_staves)-1 else (stave[4]+treble_staves[si+1][0])//2

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
                notes.append({'x': cx, 'y': cy, 'stave': si, 'pitch': pitch, 'kanji': kanji, 'suffix': suffix})

    notes.sort(key=lambda n: (n['stave'], n['x']))

    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    drawn = 0
    for n in notes:
        full = n['kanji'] + n['suffix']
        col = (180, 0, 0, 255) if not n['suffix'] else (26, 106, 170, 255) if n['suffix'] == '△' else (122, 26, 170, 255)
        stave = treble_staves[n['stave']]
        top_y = stave[0] - 8
        bbox = draw.textbbox((0, 0), full, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = n['x'] - tw//2
        ty = top_y - th
        draw.rectangle([tx-2, ty-2, tx+tw+2, ty+th+2], fill=(255, 255, 255, 210))
        draw.text((tx, ty), full, font=font, fill=col)
        drawn += 1

    result_img = Image.alpha_composite(img_pil, overlay).convert('RGB')

    # 調弦表を画像下部に追加
    result = add_tuning_table(result_img, tuning_name, font_size)

    tr_str = f'（移調{transpose:+d}半音）' if transpose != 0 else ''
    msg = f"{drawn}個の音符を検出して箏符を付与しました{tr_str}"

    return result, msg, suggested_tuning, get_tuning_display(tuning_name)

# ─────────────────────────────────────────────
# Flask ルート
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', presets=PRESET_ORDER)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '画像が必要です'}), 400
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        H, W = img_cv.shape
        _, binary = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY_INV)
        treble_staves = detect_staves(binary, W)
        if not treble_staves:
            return jsonify({'error': '五線譜が検出できませんでした'}), 400
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        hl = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
        bn = cv2.subtract(binary, hl)
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vl = cv2.morphologyEx(bn, cv2.MORPH_OPEN, vk)
        bn = cv2.subtract(bn, vl)
        bk = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        bl = cv2.morphologyEx(bn, cv2.MORPH_OPEN, bk)
        bn_clean = cv2.subtract(bn, bl)
        contours, _ = cv2.findContours(bn_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        used_midi = extract_used_pitches(contours, treble_staves, binary, W)
        suggested, unmatched = suggest_tuning(used_midi)
        tuning_display = get_tuning_display(suggested)
        used_names = sorted(set(midi_to_name(m) for m in used_midi))
        unmatched_names = sorted(set(midi_to_name(m) for m in unmatched))
        return jsonify({
            'suggested': suggested,
            'tuning_display': tuning_display,
            'used_notes': used_names,
            'unmatched_notes': unmatched_names,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        file = request.files.get('image')
        tuning = request.form.get('tuning', 'G調')
        font_size = int(request.form.get('font_size', 20))
        transpose = int(request.form.get('transpose', 0))
        auto_suggest = request.form.get('auto_suggest', 'false') == 'true'
        if not file:
            return jsonify({'error': '画像が必要です'}), 400
        img_bytes = file.read()
        result, msg, suggested, tuning_display = process_image(
            img_bytes, tuning, transpose, font_size, auto_suggest
        )
        if result is None:
            return jsonify({'error': msg}), 400
        buf = io.BytesIO()
        result.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{b64}',
            'message': msg,
            'suggested_tuning': suggested,
            'tuning_display': tuning_display,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tuning_info', methods=['GET'])
def tuning_info():
    name = request.args.get('name', 'G調')
    return jsonify({
        'name': name,
        'tuning_display': get_tuning_display(name),
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

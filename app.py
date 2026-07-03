from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

STRING_KANJI = ['一','二','三','四','五','六','七','八','九','十','斗','為','巾']
CHROMATIC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ENHARMONIC = {'Bb':'A#','Eb':'D#','Ab':'G#','Db':'C#','Gb':'F#'}

PRESETS = {
    'C調': [('C',4),('D',4),('E',4),('F',4),('G',4),('A',4),('C',5),('D',5),('E',5),('F',5),('G',5),('A',5),('C',6)],
    'G調': [('G',3),('A',3),('B',3),('C',4),('D',4),('E',4),('F#',4),('G',4),('A',4),('B',4),('C',5),('D',5),('E',5)],
    'D調': [('D',4),('E',4),('F#',4),('G',4),('A',4),('B',4),('D',5),('E',5),('F#',5),('G',5),('A',5),('B',5),('D',6)],
    'F調': [('F',3),('G',3),('A',3),('A#',3),('C',4),('D',4),('E',4),('F',4),('G',4),('A',4),('A#',4),('C',5),('D',5)],
    'Bb調':[('A#',3),('C',4),('D',4),('D#',4),('F',4),('G',4),('A',4),('A#',4),('C',5),('D',5),('D#',5),('F',5),('G',5)],
    '平調子':[('D',4),('E',4),('F#',4),('A',4),('B',4),('D',5),('E',5),('F#',5),('A',5),('B',5),('D',6),('E',6),('F#',6)],
    '雲井調子':[('D',4),('E',4),('G',4),('A',4),('B',4),('D',5),('E',5),('G',5),('A',5),('B',5),('D',6),('E',6),('G',6)],
}

PITCH_STEPS = {
    'C4':-3.5,'D4':-3,'E4':-2.5,'F4':-2,'F#4':-2,
    'G4':-1.5,'G#4':-1.5,'A4':-1,'A#4':-1,'B4':-0.5,
    'C5':0,'C#5':0,'D5':0.5,'D#5':0.5,'E5':1,
    'F5':1.5,'F#5':1.5,'G5':2,'A5':2.5,'B5':3,'C6':3.5,
}
STEP_TO_PITCH = {v:k for k,v in PITCH_STEPS.items()}

def note_to_midi(note, octave):
    n = ENHARMONIC.get(note, note)
    return (int(octave)+1)*12 + CHROMATIC.index(n)

def build_midi_map(tuning):
    m = {}
    for i,(note,octave) in enumerate(tuning):
        midi = note_to_midi(note, octave)
        m[midi] = (STRING_KANJI[i], '')
        m[midi+1] = (STRING_KANJI[i], '△')
        m[midi+2] = (STRING_KANJI[i], '▲')
    return m

def detect_staves(binary, W):
    row_sums = np.sum(binary, axis=1)/255
    cands = np.where(row_sums > W*0.35)[0]
    if len(cands)==0: return []
    grps,cur=[],[cands[0]]
    for y in cands[1:]:
        if y-cur[-1]<=4: cur.append(y)
        else: grps.append(cur); cur=[y]
    grps.append(cur)
    lc=[int(np.mean(g)) for g in grps]
    staves=[]
    i=0
    while i+4<len(lc):
        five=lc[i:i+5]
        gaps=[five[j+1]-five[j] for j in range(4)]
        if np.mean(gaps)>3 and max(gaps)-min(gaps)<np.mean(gaps)*0.6:
            staves.append(five); i+=5
        else: i+=1
    return staves[::2]  # ト音記号段のみ

def find_clef_end(stave, binary, W):
    y1,y2=stave[0]-2,stave[4]+2
    cs=np.sum(binary[y1:y2,:],axis=0)/255
    th=(y2-y1)*0.65
    vc=np.where(cs>th)[0]
    valid=vc[vc>int(W*0.06)]
    if len(valid)==0: return int(W*0.13)
    gps,cu=[],[valid[0]]
    for x in valid[1:]:
        if x-cu[-1]<=8: cu.append(x)
        else: gps.append(cu); cu=[x]
    gps.append(cu)
    return int(np.mean(gps[0]))+8

def y_to_pitch(cy, stave):
    gap=np.mean([stave[j+1]-stave[j] for j in range(4)])
    g4_y=stave[1]
    step=(g4_y-cy)/gap
    steps=list(STEP_TO_PITCH.keys())
    nearest=min(steps,key=lambda s:abs(s-step))
    return STEP_TO_PITCH[nearest]

def process_image(img_bytes, tuning_name, font_size=20):
    nparr=np.frombuffer(img_bytes,np.uint8)
    img_cv=cv2.imdecode(nparr,cv2.IMREAD_GRAYSCALE)
    H,W=img_cv.shape

    _,binary=cv2.threshold(img_cv,200,255,cv2.THRESH_BINARY_INV)
    treble_staves=detect_staves(binary,W)
    if not treble_staves:
        return None,"五線譜が検出できませんでした"

    # 五線・符幹・梁を除去
    hk=cv2.getStructuringElement(cv2.MORPH_RECT,(50,1))
    hl=cv2.morphologyEx(binary,cv2.MORPH_OPEN,hk)
    bn=cv2.subtract(binary,hl)
    vk=cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))
    vl=cv2.morphologyEx(bn,cv2.MORPH_OPEN,vk)
    bn=cv2.subtract(bn,vl)
    bk=cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
    bl=cv2.morphologyEx(bn,cv2.MORPH_OPEN,bk)
    bn_clean=cv2.subtract(bn,bl)

    contours,_=cv2.findContours(bn_clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    tuning=PRESETS.get(tuning_name,PRESETS['C調'])
    midi_map=build_midi_map(tuning)

    notes=[]
    for si,stave in enumerate(treble_staves):
        clef_end=find_clef_end(stave,binary,W)
        gap=np.mean([stave[j+1]-stave[j] for j in range(4)])
        y_top=0 if si==0 else (treble_staves[si-1][4]+stave[0])//2
        y_bot=H if si==len(treble_staves)-1 else (stave[4]+treble_staves[si+1][0])//2

        for cnt in contours:
            x,y,w,h=cv2.boundingRect(cnt)
            cx,cy=x+w//2,y+h//2
            if not(y_top<=cy<=y_bot): continue
            if cx<clef_end: continue
            area=cv2.contourArea(cnt)
            aspect=w/h if h>0 else 0
            if not(gap*0.6<=w<=gap*2.2 and gap*0.4<=h<=gap*1.6): continue
            if not(0.6<=aspect<=2.5): continue
            if area<gap*gap*0.2: continue
            pitch=y_to_pitch(cy,stave)
            midi=note_to_midi(pitch[:-1],int(pitch[-1]))
            if midi in midi_map:
                kanji,suffix=midi_map[midi]
                notes.append({'x':cx,'y':cy,'stave':si,'pitch':pitch,'kanji':kanji,'suffix':suffix})

    notes.sort(key=lambda n:(n['stave'],n['x']))

    # 画像合成
    font_paths=['/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
                '/usr/share/fonts/truetype/noto/NotoSerifCJK-Bold.ttc']
    font=None
    for fp in font_paths:
        if os.path.exists(fp):
            font=ImageFont.truetype(fp,font_size,index=0); break
    if font is None: font=ImageFont.load_default()

    img_pil=Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    overlay=Image.new('RGBA',img_pil.size,(0,0,0,0))
    draw=ImageDraw.Draw(overlay)

    drawn=0
    for n in notes:
        full=n['kanji']+n['suffix']
        col=(180,0,0,255) if not n['suffix'] else (26,106,170,255) if n['suffix']=='△' else (122,26,170,255)
        stave=treble_staves[n['stave']]
        top_y=stave[0]-8
        bbox=draw.textbbox((0,0),full,font=font)
        tw,th=bbox[2]-bbox[0],bbox[3]-bbox[1]
        tx=n['x']-tw//2
        ty=top_y-th
        draw.rectangle([tx-2,ty-2,tx+tw+2,ty+th+2],fill=(255,255,255,210))
        draw.text((tx,ty),full,font=font,fill=col)
        drawn+=1

    result=Image.alpha_composite(img_pil,overlay).convert('RGB')
    return result, f"{drawn}個の音符を自動検出して箏符を付与しました"

@app.route('/')
def index():
    return render_template('index.html', presets=list(PRESETS.keys()))

@app.route('/process', methods=['POST'])
def process():
    try:
        file=request.files.get('image')
        tuning=request.form.get('tuning','C調')
        font_size=int(request.form.get('font_size',20))
        if not file:
            return jsonify({'error':'画像が必要です'}),400
        img_bytes=file.read()
        result,msg=process_image(img_bytes,tuning,font_size)
        if result is None:
            return jsonify({'error':msg}),400
        buf=io.BytesIO()
        result.save(buf,format='JPEG',quality=92)
        buf.seek(0)
        b64=base64.b64encode(buf.read()).decode()
        return jsonify({'success':True,'image':f'data:image/jpeg;base64,{b64}','message':msg})
    except Exception as e:
        return jsonify({'error':str(e)}),500

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)

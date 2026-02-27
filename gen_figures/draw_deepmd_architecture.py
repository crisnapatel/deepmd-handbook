"""
Generate DeePMD-kit neural network architecture diagram for Ch 2.
Shows: Coordinates → Descriptor (se_e2_a) → Fitting Net → Energy/Forces.
Output: content/assets/diagrams/deepmd_architecture.svg
"""
import drawsvg as draw

d = draw.Drawing(800, 380, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 800, 380, fill='white'))

# Title
d.append(draw.Text('DeePMD-kit Architecture (se_e2_a)', 18, 400, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

y_mid = 190

# === Stage boxes ===
stages = [
    (60, 'Atomic\nCoordinates', '#ECEFF1', '#546E7A', 'R = {r₁, r₂, ...}'),
    (210, 'Local\nEnvironment', '#E3F2FD', '#1565C0', 'R̃ᵢ (neighbor list)'),
    (370, 'Descriptor\n(se_e2_a)', '#FFF3E0', '#E65100', 'Dᵢ = R̃ᵢᵀ · G(R̃ᵢ) · R̃ᵢ'),
    (530, 'Fitting\nNetwork', '#E8F5E9', '#2E7D32', '[240, 240, 240]'),
    (690, 'Energy\nForces', '#FCE4EC', '#C62828', 'E = ΣEᵢ, F = -∇E'),
]

bw, bh = 120, 70
arrow = draw.Marker(-0.5, -0.5, 0.5, 0.5, scale=7, orient='auto')
arrow.append(draw.Lines(-0.5, -0.5, -0.5, 0.5, 0.5, 0, fill='#888', close=True))

for i, (cx, label, bg, border, sublabel) in enumerate(stages):
    # Box
    d.append(draw.Rectangle(cx - bw/2, y_mid - bh/2, bw, bh, rx=8, ry=8,
                             fill=bg, stroke=border, stroke_width=2))
    # Label (handle newlines)
    lines = label.split('\n')
    for j, line in enumerate(lines):
        d.append(draw.Text(line, 13, cx, y_mid - 8 + j * 16,
                           text_anchor='middle', font_weight='bold', fill=border))
    # Sublabel
    d.append(draw.Text(sublabel, 9, cx, y_mid + bh/2 + 14,
                        text_anchor='middle', fill='#888'))

    # Arrow to next stage
    if i < len(stages) - 1:
        nx = stages[i + 1][0]
        d.append(draw.Line(cx + bw/2 + 2, y_mid, nx - bw/2 - 8, y_mid,
                           stroke='#888', stroke_width=2, marker_end=arrow))

# === Detail annotations below ===
y_detail = 280

# Descriptor details
d.append(draw.Rectangle(140, y_detail - 5, 240, 80, rx=6, ry=6,
                         fill='#FFF8E1', stroke='#FFB300', stroke_width=1.5))
d.append(draw.Text('Descriptor Parameters', 11, 260, y_detail + 10,
                    text_anchor='middle', font_weight='bold', fill='#F57F17'))
params = ['rcut = 6.0 Å (cutoff radius)',
          'rcut_smth = 2.0 Å (smooth start)',
          'sel = [60, 120] (max neighbors)',
          'neuron = [25, 50, 100] (embedding)']
for i, p in enumerate(params):
    d.append(draw.Text(p, 9, 160, y_detail + 25 + i * 14,
                        fill='#555'))

# Fitting net details
d.append(draw.Rectangle(430, y_detail - 5, 220, 80, rx=6, ry=6,
                         fill='#E8F5E9', stroke='#4CAF50', stroke_width=1.5))
d.append(draw.Text('Fitting Net Parameters', 11, 540, y_detail + 10,
                    text_anchor='middle', font_weight='bold', fill='#2E7D32'))
fparams = ['3 hidden layers: [240, 240, 240]',
           'resnet_dt = true',
           'Activation: tanh',
           'Output: per-atom energy Eᵢ']
for i, p in enumerate(fparams):
    d.append(draw.Text(p, 9, 445, y_detail + 25 + i * 14,
                        fill='#555'))

# Dashed lines connecting detail boxes to main stages
d.append(draw.Line(370, y_mid + bh/2, 260, y_detail - 5,
                   stroke='#FFB300', stroke_width=1, stroke_dasharray='4'))
d.append(draw.Line(530, y_mid + bh/2, 540, y_detail - 5,
                   stroke='#4CAF50', stroke_width=1, stroke_dasharray='4'))

# Key insight annotation
d.append(draw.Rectangle(10, 50, 200, 50, rx=6, ry=6,
                         fill='#E8EAF6', stroke='#3F51B5', stroke_width=1.5))
d.append(draw.Text('Key: Each atom gets its own', 9, 110, 68,
                    text_anchor='middle', fill='#3F51B5'))
d.append(draw.Text('descriptor → fitting net → Eᵢ', 9, 110, 82,
                    text_anchor='middle', font_weight='bold', fill='#3F51B5'))

d.save_svg('content/assets/diagrams/deepmd_architecture.svg')
print('Saved: content/assets/diagrams/deepmd_architecture.svg')

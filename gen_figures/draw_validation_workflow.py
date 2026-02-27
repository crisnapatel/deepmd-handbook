"""
Generate validation workflow diagram for Ch 11.
Shows the testing pipeline: dp test → long MD → property validation.
Output: content/assets/diagrams/validation_workflow.svg
"""
import drawsvg as draw

d = draw.Drawing(700, 400, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 700, 400, fill='white'))

# Title
d.append(draw.Text('Model Validation Pipeline', 18, 350, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

arrow = draw.Marker(-0.5, -0.5, 0.5, 0.5, scale=7, orient='auto')
arrow.append(draw.Lines(-0.5, -0.5, -0.5, 0.5, 0.5, 0, fill='#555', close=True))

# Three main stages
stages = [
    {
        'x': 120, 'y': 100,
        'title': 'dp test',
        'subtitle': 'Quantitative Baseline',
        'color': '#1565C0', 'bg': '#E3F2FD',
        'checks': [
            'Energy RMSE < 3 meV/atom',
            'Force RMSE < 60 meV/Å',
            'Per-subsystem breakdown',
        ]
    },
    {
        'x': 350, 'y': 100,
        'title': 'Long MD',
        'subtitle': 'Stability Test',
        'color': '#E65100', 'bg': '#FFF3E0',
        'checks': [
            'Run 1+ ns NVT/NPT',
            'No energy drift',
            'No atomic explosions',
        ]
    },
    {
        'x': 580, 'y': 100,
        'title': 'Properties',
        'subtitle': 'Physical Validation',
        'color': '#2E7D32', 'bg': '#E8F5E9',
        'checks': [
            'RDF matches DFT/expt',
            'Diffusion coefficient',
            'EOS / phonons',
        ]
    },
]

bw, bh = 190, 160

for i, s in enumerate(stages):
    cx, cy = s['x'], s['y']
    d.append(draw.Rectangle(cx - bw/2, cy, bw, bh, rx=8, ry=8,
                             fill=s['bg'], stroke=s['color'], stroke_width=2))
    d.append(draw.Text(s['title'], 15, cx, cy + 22, text_anchor='middle',
                        font_weight='bold', fill=s['color']))
    d.append(draw.Text(s['subtitle'], 10, cx, cy + 38, text_anchor='middle',
                        fill='#888'))

    # Divider
    d.append(draw.Line(cx - bw/2 + 10, cy + 48, cx + bw/2 - 10, cy + 48,
                       stroke=s['color'], stroke_width=0.5, opacity=0.3))

    for j, check in enumerate(s['checks']):
        yy = cy + 65 + j * 28
        d.append(draw.Circle(cx - bw/2 + 18, yy, 6, fill=s['color'], opacity=0.15))
        d.append(draw.Text('✓', 8, cx - bw/2 + 18, yy + 3, text_anchor='middle',
                            fill=s['color'], font_weight='bold'))
        d.append(draw.Text(check, 10, cx - bw/2 + 30, yy + 3, fill='#555'))

    # Arrow to next
    if i < len(stages) - 1:
        nx = stages[i + 1]['x']
        d.append(draw.Line(cx + bw/2 + 2, cy + bh/2, nx - bw/2 - 8, cy + bh/2,
                           stroke='#555', stroke_width=2, marker_end=arrow))
        # Pass/fail label
        d.append(draw.Text('Pass?', 9, (cx + bw/2 + nx - bw/2) / 2, cy + bh/2 - 10,
                            text_anchor='middle', fill='#999', font_style='italic'))

# Bottom: decision box
dy = 310
d.append(draw.Rectangle(100, dy, 500, 70, rx=8, ry=8,
                         fill='#FCE4EC', stroke='#C62828', stroke_width=1.5))
d.append(draw.Text('If any stage fails:', 12, 350, dy + 18,
                    text_anchor='middle', font_weight='bold', fill='#C62828'))

decisions = [
    'dp test fails → more training data or longer training',
    'Long MD unstable → gap-filling or adjust trust levels',
    'Properties wrong → check DFT settings, add missing configurations',
]
for i, dec in enumerate(decisions):
    d.append(draw.Text(dec, 9, 130, dy + 35 + i * 13, fill='#555'))

# Arrow from stages to decision
d.append(draw.Line(350, 260, 350, dy - 2, stroke='#C62828', stroke_width=1.5,
                   stroke_dasharray='4', marker_end=arrow))
d.append(draw.Text('Fail', 9, 360, 285, fill='#C62828', font_style='italic'))

d.save_svg('content/assets/diagrams/validation_workflow.svg')
print('Saved: content/assets/diagrams/validation_workflow.svg')

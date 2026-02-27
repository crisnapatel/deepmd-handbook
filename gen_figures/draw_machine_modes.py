"""
Generate machine.json execution modes comparison for Ch 8.
Shows: Local Shell vs HPC (PBS/Slurm) vs Mixed-mode side by side.
Output: content/assets/diagrams/machine_modes.svg
"""
import drawsvg as draw

d = draw.Drawing(800, 450, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 800, 450, fill='white'))

# Title
d.append(draw.Text('machine.json Execution Modes', 18, 400, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# Three columns
cols = [
    {
        'title': 'Local Shell',
        'subtitle': 'Single workstation / GPU node',
        'x': 140, 'color': '#1565C0', 'bg': '#E3F2FD',
        'items': [
            ('context_type', 'local'),
            ('batch_type', 'shell'),
            ('GPU', 'direct access'),
            ('Use case', 'Testing, small runs'),
        ],
        'pros': ['Simple setup', 'No queue wait'],
        'cons': ['Single machine only', 'No job scheduling'],
    },
    {
        'title': 'Full HPC',
        'subtitle': 'All stages on PBS/Slurm',
        'x': 400, 'color': '#E65100', 'bg': '#FFF3E0',
        'items': [
            ('context_type', 'ssh / local'),
            ('batch_type', 'pbs / slurm'),
            ('GPU', 'via job script'),
            ('Use case', 'Production runs'),
        ],
        'pros': ['Scalable', 'Proper resource mgmt'],
        'cons': ['Queue wait times', 'Complex config'],
    },
    {
        'title': 'Mixed-Mode',
        'subtitle': 'Shell for GPU, HPC for DFT',
        'x': 660, 'color': '#2E7D32', 'bg': '#E8F5E9',
        'items': [
            ('train', 'shell (GPU node)'),
            ('model_devi', 'shell (GPU node)'),
            ('fp', 'pbs (CPU cluster)'),
            ('Use case', 'Best of both worlds'),
        ],
        'pros': ['No queue for GPU tasks', 'HPC for heavy DFT'],
        'cons': ['More complex config', 'Need SSH between'],
    },
]

card_w, card_h = 220, 360
y_top = 50

for col in cols:
    cx = col['x']
    color = col['color']
    bg = col['bg']

    # Card background
    d.append(draw.Rectangle(cx - card_w/2, y_top, card_w, card_h, rx=10, ry=10,
                             fill=bg, stroke=color, stroke_width=2))

    # Title
    d.append(draw.Text(col['title'], 16, cx, y_top + 25,
                        text_anchor='middle', font_weight='bold', fill=color))
    d.append(draw.Text(col['subtitle'], 9, cx, y_top + 42,
                        text_anchor='middle', fill='#888'))

    # Divider
    d.append(draw.Line(cx - card_w/2 + 15, y_top + 55, cx + card_w/2 - 15, y_top + 55,
                       stroke=color, stroke_width=0.5, opacity=0.4))

    # Config items
    y = y_top + 75
    for key, val in col['items']:
        d.append(draw.Text(key + ':', 10, cx - card_w/2 + 15, y,
                            fill='#555', font_weight='bold'))
        d.append(draw.Text(val, 10, cx - card_w/2 + 15, y + 14,
                            fill=color, font_family='monospace'))
        y += 35

    # Divider
    d.append(draw.Line(cx - card_w/2 + 15, y, cx + card_w/2 - 15, y,
                       stroke=color, stroke_width=0.5, opacity=0.4))
    y += 15

    # Pros
    d.append(draw.Text('✓ Pros', 10, cx - card_w/2 + 15, y,
                        fill='#2E7D32', font_weight='bold'))
    y += 16
    for pro in col['pros']:
        d.append(draw.Text('• ' + pro, 9, cx - card_w/2 + 20, y, fill='#555'))
        y += 14

    y += 8

    # Cons
    d.append(draw.Text('✗ Cons', 10, cx - card_w/2 + 15, y,
                        fill='#C62828', font_weight='bold'))
    y += 16
    for con in col['cons']:
        d.append(draw.Text('• ' + con, 9, cx - card_w/2 + 20, y, fill='#555'))
        y += 14

# Recommendation arrow pointing to Mixed-Mode
d.append(draw.Text('★ Recommended for research workflows', 11, 400, 430,
                    text_anchor='middle', fill='#2E7D32', font_weight='bold'))

d.save_svg('content/assets/diagrams/machine_modes.svg')
print('Saved: content/assets/diagrams/machine_modes.svg')

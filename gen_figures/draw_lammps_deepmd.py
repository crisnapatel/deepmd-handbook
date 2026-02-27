"""
Generate LAMMPS + DeePMD workflow diagram for Ch 5.
Shows how the frozen model plugs into LAMMPS.
Output: content/assets/diagrams/lammps_deepmd.svg
"""
import drawsvg as draw

d = draw.Drawing(700, 350, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 700, 350, fill='white'))

# Title
d.append(draw.Text('LAMMPS with pair_style deepmd', 18, 350, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# Arrow marker
arrow = draw.Marker(-0.5, -0.5, 0.5, 0.5, scale=7, orient='auto')
arrow.append(draw.Lines(-0.5, -0.5, -0.5, 0.5, 0.5, 0, fill='#555', close=True))

y_mid = 140
bw, bh = 130, 65

# Frozen model
d.append(draw.Rectangle(30, y_mid - bh/2, bw, bh, rx=8, ry=8,
                         fill='#E8F5E9', stroke='#2E7D32', stroke_width=2))
d.append(draw.Text('graph.pb', 14, 95, y_mid - 8, text_anchor='middle',
                    font_weight='bold', fill='#2E7D32', font_family='monospace'))
d.append(draw.Text('Frozen model', 10, 95, y_mid + 10, text_anchor='middle', fill='#666'))
d.append(draw.Text('(from dp freeze)', 9, 95, y_mid + 23, text_anchor='middle', fill='#999'))

# Arrow
d.append(draw.Line(160, y_mid, 200, y_mid, stroke='#555', stroke_width=2, marker_end=arrow))

# LAMMPS box
d.append(draw.Rectangle(210, y_mid - bh/2 - 15, bw + 40, bh + 30, rx=8, ry=8,
                         fill='#E3F2FD', stroke='#1565C0', stroke_width=2))
d.append(draw.Text('LAMMPS', 16, 300, y_mid - 18, text_anchor='middle',
                    font_weight='bold', fill='#1565C0'))
d.append(draw.Text('pair_style deepmd', 11, 300, y_mid, text_anchor='middle',
                    fill='#1565C0', font_family='monospace'))
d.append(draw.Text('positions → model → forces', 9, 300, y_mid + 18,
                    text_anchor='middle', fill='#666'))
d.append(draw.Text('(every timestep)', 9, 300, y_mid + 32, text_anchor='middle', fill='#999'))

# Arrow to outputs
d.append(draw.Line(390, y_mid, 430, y_mid, stroke='#555', stroke_width=2, marker_end=arrow))

# Output box
d.append(draw.Rectangle(440, y_mid - bh/2 - 25, bw + 80, bh + 50, rx=8, ry=8,
                         fill='#FFF3E0', stroke='#E65100', stroke_width=2))
d.append(draw.Text('MD Trajectory', 14, 530, y_mid - 25, text_anchor='middle',
                    font_weight='bold', fill='#E65100'))

outputs = [
    ('dump.lammpstrj', 'positions, velocities'),
    ('log.lammps', 'thermo output'),
    ('md.out', 'model deviation'),
]
for i, (name, desc) in enumerate(outputs):
    yy = y_mid - 5 + i * 20
    d.append(draw.Text(name, 10, 460, yy, fill='#E65100',
                        font_family='monospace', font_weight='bold'))
    d.append(draw.Text(desc, 8, 555, yy, fill='#999'))

# Bottom: what pair_style deepmd does
by = 240
d.append(draw.Rectangle(50, by, 600, 90, rx=8, ry=8,
                         fill='#FAFAFA', stroke='#DDD', stroke_width=1.5))
d.append(draw.Text('What happens at each timestep:', 11, 350, by + 16,
                    text_anchor='middle', font_weight='bold', fill='#333'))

steps = [
    ('1', 'LAMMPS sends atomic positions to the NN', '#1565C0'),
    ('2', 'NN computes per-atom energies and forces', '#2E7D32'),
    ('3', 'LAMMPS uses forces to integrate equations of motion', '#E65100'),
    ('4', 'Repeat (no QM calculation needed)', '#7B1FA2'),
]
for i, (num, text, color) in enumerate(steps):
    yy = by + 32 + i * 15
    d.append(draw.Circle(70, yy - 2, 8, fill=color, opacity=0.15))
    d.append(draw.Text(num, 9, 70, yy + 1, text_anchor='middle',
                        font_weight='bold', fill=color))
    d.append(draw.Text(text, 10, 85, yy, fill='#555'))

d.save_svg('content/assets/diagrams/lammps_deepmd.svg')
print('Saved: content/assets/diagrams/lammps_deepmd.svg')

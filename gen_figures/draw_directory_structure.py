"""
Generate dpgen iteration directory structure diagram for Ch 9.
Shows the tree layout: iter.000000/{00.train, 01.model_devi, 02.fp}
Output: content/assets/diagrams/directory_structure.svg
"""
import drawsvg as draw

d = draw.Drawing(750, 620, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 750, 620, fill='white'))

# Title
d.append(draw.Text('dpgen Iteration Directory Structure', 18, 375, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

x0, y0 = 40, 55
line_h = 22
indent = 24
font_s = 12

def draw_entry(x, y, text, is_dir=True, color='#333', note=''):
    """Draw a tree entry with optional annotation."""
    icon_color = '#FFB300' if is_dir else '#78909C'
    icon = '📁' if is_dir else '📄'
    # Use a small colored rectangle as icon instead of emoji for SVG compat
    if is_dir:
        d.append(draw.Rectangle(x, y - 8, 12, 10, rx=2, ry=2,
                                fill='#FFE082', stroke='#FFB300', stroke_width=1))
    else:
        d.append(draw.Rectangle(x, y - 8, 10, 12, rx=1, ry=1,
                                fill='#E0E0E0', stroke='#9E9E9E', stroke_width=1))
    d.append(draw.Text(text, font_s, x + 16, y + 2, fill=color,
                        font_family='monospace'))
    if note:
        d.append(draw.Text(note, 9, x + 16 + len(text) * 7.5, y + 2,
                            fill='#999', font_style='italic'))

# Tree connector lines
def draw_branch(x, y_start, y_end):
    d.append(draw.Line(x + 5, y_start, x + 5, y_end, stroke='#CCC', stroke_width=1.5))

def draw_connector(x, y):
    d.append(draw.Lines(x + 5, y - line_h/2, x + 5, y, x + 14, y,
                        stroke='#CCC', stroke_width=1.5, fill='none'))

y = y0

# Root
draw_entry(x0, y, 'dpgen_workdir/', color='#333', note='  (your dpgen run directory)')
y += line_h

# iter.000000
draw_connector(x0 + indent, y)
draw_entry(x0 + indent, y, 'iter.000000/', color='#1565C0', note='  ← Iteration 0')
y += line_h

# 00.train
i2 = x0 + 2 * indent
draw_connector(i2, y)
draw_entry(i2, y, '00.train/', color='#1565C0')
y += line_h

i3 = x0 + 3 * indent
for f, note in [
    ('000/', '  ← Model 0 (seed 1)'),
    ('001/', '  ← Model 1 (seed 2)'),
    ('002/', '  ← Model 2 (seed 3)'),
    ('003/', '  ← Model 3 (seed 4)'),
]:
    draw_connector(i3, y)
    draw_entry(i3, y, f, color='#42A5F5', note=note)
    y += line_h
    if f == '000/':
        i4 = x0 + 4 * indent
        for sf, sn in [
            ('input.json', '  training config'),
            ('lcurve.out', '  loss vs step'),
            ('model.ckpt.*', '  checkpoints'),
            ('frozen_model.pb', '  final model'),
        ]:
            draw_connector(i4, y)
            draw_entry(i4, y, sf, is_dir=False, color='#666', note=sn)
            y += line_h

# 01.model_devi
y += 5
draw_connector(i2, y)
draw_entry(i2, y, '01.model_devi/', color='#E65100')
y += line_h

for f, note in [
    ('task.000.000000/', '  ← sys_idx=0, T=77K'),
    ('task.000.000001/', '  ← sys_idx=0, T=150K'),
    ('task.001.000000/', '  ← sys_idx=1, T=77K'),
]:
    draw_connector(i3, y)
    draw_entry(i3, y, f, color='#FF8A65', note=note)
    y += line_h
    if 'task.000.000000' in f:
        for sf, sn in [
            ('input.lammps', '  LAMMPS input'),
            ('model_devi.out', '  deviation per frame'),
            ('traj/', '  MD trajectory'),
        ]:
            draw_connector(i4, y)
            draw_entry(i4, y, sf, is_dir=('/' in sf), color='#666', note=sn)
            y += line_h

# 02.fp
y += 5
draw_connector(i2, y)
draw_entry(i2, y, '02.fp/', color='#2E7D32')
y += line_h

for f, note in [
    ('task.000.000000/', '  ← candidate structure 1'),
    ('task.000.000001/', '  ← candidate structure 2'),
    ('candidate.shuffled.000.out', '  selection log'),
    ('data.000/', '  ← converted dpdata output'),
]:
    draw_connector(i3, y)
    is_d = '/' in f
    draw_entry(i3, y, f, is_dir=is_d, color='#66BB6A' if is_d else '#666', note=note)
    y += line_h
    if 'task.000.000000' in f:
        for sf, sn in [
            ('input', '  QE/VASP input file'),
            ('output', '  DFT output'),
            ('POSCAR', '  structure'),
        ]:
            draw_connector(i4, y)
            draw_entry(i4, y, sf, is_dir=False, color='#666', note=sn)
            y += line_h

# iter.000001 (collapsed)
y += 10
draw_connector(x0 + indent, y)
draw_entry(x0 + indent, y, 'iter.000001/', color='#1565C0', note='  ← Iteration 1 (same structure)')
y += line_h

# ...
d.append(draw.Text('...', 14, x0 + indent + 16, y + 2, fill='#999',
                    font_family='monospace'))
y += line_h

# Key files at root
draw_connector(x0 + indent, y)
draw_entry(x0 + indent, y, 'record.dpgen', is_dir=False, color='#C62828',
           note='  ← state machine (tracks progress)')
y += line_h
draw_connector(x0 + indent, y)
draw_entry(x0 + indent, y, 'dpgen.log', is_dir=False, color='#C62828',
           note='  ← main log file')

d.save_svg('content/assets/diagrams/directory_structure.svg')
print('Saved: content/assets/diagrams/directory_structure.svg')

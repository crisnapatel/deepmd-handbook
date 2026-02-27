"""
Generate dpdata file format diagram for Ch 3.
Shows the directory structure and what each file contains.
Output: content/assets/diagrams/data_format.svg
"""
import drawsvg as draw

d = draw.Drawing(700, 480, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 700, 480, fill='white'))

# Title
d.append(draw.Text('DeePMD-kit Data Format (npy)', 18, 350, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# Left side: directory tree
x0, y0 = 30, 55
line_h = 24
fs = 12

def draw_item(x, y, name, is_dir=True, desc=''):
    if is_dir:
        d.append(draw.Rectangle(x, y - 9, 13, 11, rx=2, ry=2,
                                fill='#FFE082', stroke='#FFB300', stroke_width=1))
    else:
        d.append(draw.Rectangle(x, y - 9, 11, 13, rx=1, ry=1,
                                fill='#E0E0E0', stroke='#9E9E9E', stroke_width=1))
    d.append(draw.Text(name, fs, x + 18, y + 2, fill='#333', font_family='monospace'))
    if desc:
        d.append(draw.Text(desc, 9, x + 18 + len(name) * 7.5, y + 2,
                           fill='#999', font_style='italic'))

y = y0
draw_item(x0, y, 'my_system/', desc='')
y += line_h
draw_item(x0 + 24, y, 'type.raw', False, '  0 0 0 ... 1 1')
y += line_h
draw_item(x0 + 24, y, 'type_map.raw', False, '  C  H')
y += line_h
draw_item(x0 + 24, y, 'nopbc', False, '  (optional)')
y += line_h
draw_item(x0 + 24, y, 'set.000/')
y += line_h
draw_item(x0 + 48, y, 'box.npy', False, '  (nframes, 9)')
y += line_h
draw_item(x0 + 48, y, 'coord.npy', False, '  (nframes, natoms*3)')
y += line_h
draw_item(x0 + 48, y, 'energy.npy', False, '  (nframes,)')
y += line_h
draw_item(x0 + 48, y, 'force.npy', False, '  (nframes, natoms*3)')
y += line_h
draw_item(x0 + 48, y, 'virial.npy', False, '  (nframes, 9)')
y += line_h
draw_item(x0 + 24, y, 'set.001/', desc='  (more data splits)')

# Right side: what each file means
rx = 380
ry = 55
box_w = 290
item_h = 55

items = [
    ('type.raw', 'Integer per atom (0=C, 1=H).\nSame for ALL frames. Set once.', '#E3F2FD', '#1565C0'),
    ('type_map.raw', 'Element names, one per line.\nMaps 0→C, 1→H, etc.', '#E3F2FD', '#1565C0'),
    ('box.npy', 'Cell vectors (a1,a2,a3) flattened.\n9 numbers per frame, in Angstrom.', '#FFF3E0', '#E65100'),
    ('coord.npy', 'Cartesian coordinates, flattened.\nnatoms × 3 numbers per frame, in Å.', '#FFF3E0', '#E65100'),
    ('energy.npy', 'Total energy per frame.\nIn eV (not Ry, not Ha).', '#E8F5E9', '#2E7D32'),
    ('force.npy', 'Forces on each atom, flattened.\nnatoms × 3 per frame, in eV/Å.', '#E8F5E9', '#2E7D32'),
]

for i, (name, desc, bg, border) in enumerate(items):
    yy = ry + i * (item_h + 8)
    d.append(draw.Rectangle(rx, yy, box_w, item_h, rx=6, ry=6,
                             fill=bg, stroke=border, stroke_width=1.5))
    d.append(draw.Text(name, 11, rx + 8, yy + 16,
                        font_weight='bold', fill=border, font_family='monospace'))
    lines = desc.split('\n')
    for j, line in enumerate(lines):
        d.append(draw.Text(line, 9, rx + 8, yy + 30 + j * 13, fill='#555'))

# Bottom: conversion flow
by = 430
d.append(draw.Rectangle(30, by, 640, 40, rx=8, ry=8,
                         fill='#F3E5F5', stroke='#7B1FA2', stroke_width=1.5))
d.append(draw.Text('QE output  →  dpdata.LabeledSystem()  →  .to_deepmd_npy()  →  set.000/*.npy',
                    12, 350, by + 22, text_anchor='middle', fill='#7B1FA2',
                    font_family='monospace', font_weight='bold'))

d.save_svg('content/assets/diagrams/data_format.svg')
print('Saved: content/assets/diagrams/data_format.svg')

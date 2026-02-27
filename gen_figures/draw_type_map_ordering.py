"""
Generate type_map ordering consistency diagram for Practical: Multi-Element.
Shows how element ordering must match across all files.
Output: content/assets/diagrams/type_map_ordering.svg
"""
import drawsvg as draw

d = draw.Drawing(700, 380, origin=(0, 0))
d.append(draw.Rectangle(0, 0, 700, 380, fill='white'))

# Title
d.append(draw.Text('type_map Must Match Everywhere', 18, 350, 25,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# Central type_map
cx, cy = 350, 80
bw, bh = 200, 50
d.append(draw.Rectangle(cx - bw/2, cy - bh/2, bw, bh, rx=8, ry=8,
                         fill='#E3F2FD', stroke='#1565C0', stroke_width=2.5))
d.append(draw.Text('type_map: ["C", "O", "H"]', 13, cx, cy + 4,
                    text_anchor='middle', font_weight='bold', fill='#1565C0',
                    font_family='monospace'))

arrow = draw.Marker(-0.5, -0.5, 0.5, 0.5, scale=6, orient='auto')
arrow.append(draw.Lines(-0.5, -0.5, -0.5, 0.5, 0.5, 0, fill='#888', close=True))

# Files that must match
files = [
    {'x': 100, 'y': 200, 'name': 'param.json', 'detail': '"type_map": ["C","O","H"]',
     'color': '#1565C0'},
    {'x': 350, 'y': 180, 'name': 'type.raw', 'detail': '0 0 0...1 1...2 2 2',
     'color': '#E65100'},
    {'x': 600, 'y': 200, 'name': 'type_map.raw', 'detail': 'C\\nO\\nH',
     'color': '#E65100'},
    {'x': 100, 'y': 300, 'name': 'POSCAR', 'detail': 'C  O  H  (element line)',
     'color': '#7B1FA2'},
    {'x': 350, 'y': 320, 'name': 'fp_pp_files', 'detail': '["C.upf", "O.upf", "H.upf"]',
     'color': '#2E7D32'},
    {'x': 600, 'y': 300, 'name': 'sel', 'detail': '[48, 48, 56]  (C, O, H)',
     'color': '#2E7D32'},
]

fw, fh = 170, 55

for f in files:
    d.append(draw.Rectangle(f['x'] - fw/2, f['y'] - fh/2, fw, fh, rx=6, ry=6,
                             fill='#FAFAFA', stroke=f['color'], stroke_width=1.5))
    d.append(draw.Text(f['name'], 12, f['x'], f['y'] - 8,
                        text_anchor='middle', font_weight='bold', fill=f['color'],
                        font_family='monospace'))
    d.append(draw.Text(f['detail'], 9, f['x'], f['y'] + 10,
                        text_anchor='middle', fill='#666', font_family='monospace'))

    # Arrow from central type_map
    d.append(draw.Line(cx, cy + bh/2, f['x'], f['y'] - fh/2,
                       stroke='#CCC', stroke_width=1.5, stroke_dasharray='4'))

# Warning at bottom
d.append(draw.Rectangle(120, 355, 460, 22, rx=4, ry=4,
                         fill='#FFEBEE', stroke='#F44336', stroke_width=1))
d.append(draw.Text('Mismatch in ANY of these = model trains on garbage (silently)',
                    10, 350, 369, text_anchor='middle', fill='#C62828', font_weight='bold'))

d.save_svg('content/assets/diagrams/type_map_ordering.svg')
print('Saved: content/assets/diagrams/type_map_ordering.svg')

"""
Generate dpgen 3-stage loop diagram for Ch 6.
Shows: Train → Explore → Label cycle with model deviation selection.
Output: content/assets/diagrams/dpgen_loop.svg

Coordinate system: origin='center', so (0,0) is center of canvas.
y increases DOWNWARD in SVG. Rectangle(x, y, w, h) draws from top-left corner.
"""
import drawsvg as draw

d = draw.Drawing(720, 550, origin='center')

# Background
d.append(draw.Rectangle(-360, -275, 720, 550, fill='white'))

# Title
d.append(draw.Text('The DP-GEN Active Learning Loop', 20, 0, -235,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# === Three main boxes ===
box_w, box_h = 160, 80
r = 8

# Train box (top center)
# Top-left corner at (-80, -190), so box spans x=[-80,80], y=[-190,-110]
# Center: (0, -150). Right edge center: (80, -150). Left edge center: (-80, -150)
# Bottom edge center: (0, -110)
train_x, train_y = -box_w/2, -190  # top-left corner
d.append(draw.Rectangle(train_x, train_y, box_w, box_h, rx=r, ry=r,
                         fill='#E3F2FD', stroke='#1565C0', stroke_width=2.5))
d.append(draw.Text('01.train', 11, 0, -170, text_anchor='middle', fill='#666'))
d.append(draw.Text('Train 4 Models', 16, 0, -150, text_anchor='middle',
                    font_weight='bold', fill='#1565C0'))
d.append(draw.Text('(different random seeds)', 10, 0, -130,
                    text_anchor='middle', fill='#666'))

# Explore box (bottom-right)
# Top-left corner at (110, 0), so box spans x=[110,270], y=[0,80]
# Center: (190, 40). Left edge center: (110, 40). Top edge center: (190, 0)
exp_x, exp_y = 110, 0  # top-left corner
d.append(draw.Rectangle(exp_x, exp_y, box_w, box_h, rx=r, ry=r,
                         fill='#FFF3E0', stroke='#E65100', stroke_width=2.5))
d.append(draw.Text('02.model_devi', 11, 190, 20, text_anchor='middle', fill='#666'))
d.append(draw.Text('Explore (LAMMPS)', 16, 190, 40, text_anchor='middle',
                    font_weight='bold', fill='#E65100'))
d.append(draw.Text('Run MD, measure deviation', 10, 190, 60,
                    text_anchor='middle', fill='#666'))

# Label box (bottom-left)
# Top-left corner at (-270, 0), so box spans x=[-270,-110], y=[0,80]
# Center: (-190, 40). Right edge center: (-110, 40). Top edge center: (-190, 0)
lab_x, lab_y = -270, 0  # top-left corner
d.append(draw.Rectangle(lab_x, lab_y, box_w, box_h, rx=r, ry=r,
                         fill='#E8F5E9', stroke='#2E7D32', stroke_width=2.5))
d.append(draw.Text('03.fp', 11, -190, 20, text_anchor='middle', fill='#666'))
d.append(draw.Text('Label (DFT)', 16, -190, 40, text_anchor='middle',
                    font_weight='bold', fill='#2E7D32'))
d.append(draw.Text('QE / VASP on candidates', 10, -190, 60,
                    text_anchor='middle', fill='#666'))

# === Sleek arrowhead ===
arrow = draw.Marker(-1, -0.4, 0.2, 0.4, scale=6, orient='auto')
arrow.append(draw.Lines(-1, -0.4, 0.2, 0, -1, 0.4, fill='#555', close=True))

# === Arrows (all endpoints on box edges, not inside boxes) ===

# 1. Train → Explore
#    Start: right edge of Train, center = (80, -150)
#    End: top edge of Explore, center = (190, 0)
p1 = draw.Path(stroke='#555', stroke_width=2, fill='none', marker_end=arrow)
p1.M(80, -150)
p1.C(160, -150,   # control 1: right of Train
     190, -60,    # control 2: above Explore
     190, 0)      # end: top of Explore
d.append(p1)

# 2. Explore → Label
#    Start: left edge of Explore, slightly below center = (110, 50)
#    End: right edge of Label = (-110, 50)
p2 = draw.Path(stroke='#555', stroke_width=2, fill='none', marker_end=arrow)
p2.M(110, 50)
p2.L(-110, 50)
d.append(p2)

# Label for bottom arrow
d.append(draw.Text('Select candidates', 11, 0, 70,
                    text_anchor='middle', fill='#555', font_style='italic'))
d.append(draw.Text('(trust_lo < σ_f < trust_hi)', 10, 0, 84,
                    text_anchor='middle', fill='#888'))

# 3. Label → Train
#    Start: top edge of Label, center = (-190, 0)
#    End: left edge of Train, center = (-80, -150)
p3 = draw.Path(stroke='#555', stroke_width=2, fill='none', marker_end=arrow)
p3.M(-190, 0)
p3.C(-190, -60,    # control 1: above Label
     -160, -150,   # control 2: left of Train
     -80, -150)    # end: left edge of Train
d.append(p3)

# "Repeat" label
d.append(draw.Text('Repeat until convergence', 12, -100, -60,
                    text_anchor='middle', fill='#999', font_style='italic'))

# === Three-bucket classification ===
bx, by = 0, 140
d.append(draw.Rectangle(bx - 260, by, 520, 95, rx=6, ry=6,
                         fill='#FAFAFA', stroke='#DDD', stroke_width=1.5))
d.append(draw.Text('Model Deviation Classification', 14, bx, by + 18,
                    text_anchor='middle', font_weight='bold', fill='#333'))

# Accurate
d.append(draw.Rectangle(bx - 240, by + 30, 150, 50, rx=4, ry=4,
                         fill='#C8E6C9', stroke='#4CAF50', stroke_width=1.5))
d.append(draw.Text('σ_f < trust_lo', 12, bx - 165, by + 50,
                    text_anchor='middle', fill='#2E7D32'))
d.append(draw.Text('ACCURATE', 12, bx - 165, by + 67,
                    text_anchor='middle', font_weight='bold', fill='#2E7D32'))

# Candidate
d.append(draw.Rectangle(bx - 75, by + 30, 150, 50, rx=4, ry=4,
                         fill='#FFE0B2', stroke='#FF9800', stroke_width=1.5))
d.append(draw.Text('trust_lo ≤ σ_f < trust_hi', 11, bx, by + 50,
                    text_anchor='middle', fill='#E65100'))
d.append(draw.Text('CANDIDATE → DFT', 11, bx, by + 67,
                    text_anchor='middle', font_weight='bold', fill='#E65100'))

# Failed
d.append(draw.Rectangle(bx + 90, by + 30, 150, 50, rx=4, ry=4,
                         fill='#FFCDD2', stroke='#F44336', stroke_width=1.5))
d.append(draw.Text('σ_f ≥ trust_hi', 12, bx + 165, by + 50,
                    text_anchor='middle', fill='#C62828'))
d.append(draw.Text('FAILED → skip', 12, bx + 165, by + 67,
                    text_anchor='middle', font_weight='bold', fill='#C62828'))

d.save_svg('content/assets/diagrams/dpgen_loop.svg')
print('Saved: content/assets/diagrams/dpgen_loop.svg')

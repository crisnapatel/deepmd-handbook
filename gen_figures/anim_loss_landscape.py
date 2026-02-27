"""
Manim animation: "Learn the landscape first, then adjust the elevation."

Shows how DeePMD training works in two phases:
  Phase 1 (forces dominate): Model learns the PES shape (curvature/slopes)
          but at the wrong absolute energy.
  Phase 2 (energy catches up): Model shifts vertically to match
          the true absolute energy scale.

Output: content/assets/animations/loss_landscape.mp4

Usage:
  conda run -n Analysis manim -pql gen_figures/anim_loss_landscape.py LossLandscape
  # -pql = preview, quality low (fast render). Use -pqh for high quality.
"""
from manim import *
import numpy as np


class LossLandscape(Scene):
    def construct(self):
        # ── Colors ──
        TRUE_COLOR = BLUE
        MODEL_COLOR = "#FF5722"  # orange-red
        FORCE_COLOR = "#2196F3"  # blue
        ENERGY_COLOR = "#FF5722"

        # ── Axes ──
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 4, 1],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": False, "include_tip": True},
            x_axis_config={"stroke_width": 2},
            y_axis_config={"stroke_width": 2},
        ).shift(DOWN * 0.3)

        x_label = axes.get_x_axis_label(
            Tex("Atomic Configuration", font_size=28), edge=DOWN, direction=DOWN
        )
        y_label = axes.get_y_axis_label(
            MathTex("E", font_size=32), edge=LEFT, direction=LEFT
        )

        # ── True PES ──
        def true_pes(x):
            return 0.4 * x**3 - 1.5 * x + 1.0

        true_curve = axes.plot(true_pes, x_range=[-2.5, 2.5], color=TRUE_COLOR, stroke_width=3)
        true_label = Text("True PES (DFT)", font_size=22, color=TRUE_COLOR).next_to(
            axes.c2p(1.8, true_pes(1.8)), UP, buff=0.3
        )

        # ── Phase 0: Initial model (flat / random) ──
        def initial_model(x):
            return 1.0  # flat line, no clue

        initial_curve = axes.plot(
            initial_model, x_range=[-2.5, 2.5], color=MODEL_COLOR,
            stroke_width=3, stroke_opacity=0.8
        )
        model_label = Text("Model prediction", font_size=22, color=MODEL_COLOR).next_to(
            axes.c2p(-1.5, initial_model(-1.5)), UP, buff=0.3
        )

        # ── Title ──
        title = Text("How DeePMD Learns a Potential Energy Surface", font_size=30, weight=BOLD)
        title.to_edge(UP, buff=0.3)

        # ── Scene 1: Setup ──
        self.play(Write(title), run_time=1)
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1.5)
        self.play(Create(true_curve), FadeIn(true_label), run_time=1.5)
        self.wait(0.5)

        # Show initial flat model
        step_text = Text("Step 0: Untrained model", font_size=24, color=GREY).to_edge(DOWN, buff=0.4)
        self.play(Create(initial_curve), FadeIn(model_label), FadeIn(step_text), run_time=1.5)
        self.wait(1)

        # ── Phase 1: Forces dominate — learn the shape ──
        # Model gets the curvature right but offset vertically
        VERTICAL_OFFSET = 1.5  # wrong absolute energy

        def shape_correct_model(x):
            return true_pes(x) + VERTICAL_OFFSET

        shape_curve = axes.plot(
            shape_correct_model, x_range=[-2.5, 2.5], color=MODEL_COLOR,
            stroke_width=3, stroke_opacity=0.8
        )

        phase1_text = Text(
            "Phase 1: Forces dominate (p_f = 1000)",
            font_size=24, color=FORCE_COLOR
        ).to_edge(DOWN, buff=0.4)

        # Subtitle explaining what's happening
        phase1_sub = Text(
            "Shape is correct — slopes match — but absolute energy is wrong",
            font_size=20, color=GREY_B
        ).next_to(phase1_text, UP, buff=0.15)

        new_model_label = Text("Model prediction", font_size=22, color=MODEL_COLOR).next_to(
            axes.c2p(-1.5, shape_correct_model(-1.5)), UP, buff=0.3
        )

        self.play(
            FadeOut(step_text),
            FadeIn(phase1_text),
            FadeIn(phase1_sub),
            run_time=0.8,
        )
        self.play(
            Transform(initial_curve, shape_curve),
            Transform(model_label, new_model_label),
            run_time=2.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # Show force arrows (slopes) match on both curves
        # Pick a few x-positions and draw tangent arrows
        force_arrows = VGroup()
        for x_val in [-1.5, 0.0, 1.5]:
            # Slope at this point
            dx = 0.01
            slope = (true_pes(x_val + dx) - true_pes(x_val - dx)) / (2 * dx)
            # Force = -dE/dx, show as arrow on both curves
            arrow_len = 0.6
            for func, y_offset_label in [(true_pes, 0), (shape_correct_model, 0)]:
                start = axes.c2p(x_val, func(x_val))
                # Arrow pointing in -slope direction (force direction)
                angle = np.arctan(-slope)
                end = np.array(start) + arrow_len * np.array([np.cos(angle), np.sin(angle), 0])
                arrow = Arrow(
                    start, end, buff=0, color=YELLOW,
                    stroke_width=3, max_tip_length_to_length_ratio=0.25
                )
                force_arrows.add(arrow)

        forces_label = Text("Forces (slopes) match!", font_size=22, color=YELLOW).next_to(
            axes, RIGHT, buff=0.1
        ).shift(UP * 0.5)

        self.play(
            *[GrowArrow(a) for a in force_arrows],
            FadeIn(forces_label),
            run_time=1.5,
        )
        self.wait(1.5)

        # Show the vertical gap
        gap_line = DashedLine(
            axes.c2p(0, true_pes(0)),
            axes.c2p(0, shape_correct_model(0)),
            color=RED, stroke_width=2, dash_length=0.1,
        )
        gap_label = Text("Energy offset", font_size=18, color=RED).next_to(gap_line, RIGHT, buff=0.15)
        self.play(Create(gap_line), FadeIn(gap_label), run_time=1)
        self.wait(1)

        # ── Phase 2: Energy catches up — correct the elevation ──
        def final_model(x):
            return true_pes(x)  # now matches exactly

        final_curve = axes.plot(
            final_model, x_range=[-2.5, 2.5], color=MODEL_COLOR,
            stroke_width=3, stroke_opacity=0.8
        )

        phase2_text = Text(
            "Phase 2: Energy catches up (p_e = 2)",
            font_size=24, color=ENERGY_COLOR
        ).to_edge(DOWN, buff=0.4)

        phase2_sub = Text(
            "Absolute energy scale corrected — model matches DFT",
            font_size=20, color=GREY_B
        ).next_to(phase2_text, UP, buff=0.15)

        final_model_label = Text("Model prediction", font_size=22, color=MODEL_COLOR).next_to(
            axes.c2p(-1.5, true_pes(-1.5)), DOWN, buff=0.3
        )

        self.play(
            FadeOut(force_arrows),
            FadeOut(forces_label),
            FadeOut(gap_label),
            FadeOut(phase1_text),
            FadeOut(phase1_sub),
            FadeIn(phase2_text),
            FadeIn(phase2_sub),
            run_time=0.8,
        )
        self.play(
            Transform(initial_curve, final_curve),
            Transform(model_label, final_model_label),
            FadeOut(gap_line),
            run_time=2.5,
            rate_func=smooth,
        )
        self.wait(0.5)

        # Final celebration — curves overlap
        match_label = Text(
            "Model ≈ DFT", font_size=28, color=GREEN, weight=BOLD
        ).next_to(axes, RIGHT, buff=0.2)

        self.play(FadeIn(match_label), run_time=1)
        self.wait(2)

        # ── Fade out ──
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)

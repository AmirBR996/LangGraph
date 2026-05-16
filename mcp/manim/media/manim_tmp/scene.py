
from manim import *
import numpy as np

class SelfAttentionExplainer(Scene):
    def construct(self):
        self.camera.background_color = "#0f1117"

        C_TOKEN   = "#4A9EFF"
        C_QUERY   = "#FF6B9D"
        C_KEY     = "#FFB347"
        C_VALUE   = "#50C878"
        C_ATTN    = "#C678DD"
        C_OUTPUT  = "#56B6C2"
        C_BORDER  = "#3a3f5c"
        WHITE     = "#E8EAF6"
        MUTED     = "#7A8099"

        tokens = ["The", "cat", "sat"]
        n = len(tokens)

        def make_box(text, color, width=1.1, height=0.55, font_size=22):
            rect = RoundedRectangle(corner_radius=0.08,
                                    width=width, height=height,
                                    fill_color=color, fill_opacity=0.18,
                                    stroke_color=color, stroke_width=2)
            label = Text(text, font_size=font_size, color=color)
            label.move_to(rect)
            return VGroup(rect, label)

        def make_matrix_col(header, rows, color, col_width=0.9, row_h=0.42):
            cells = VGroup()
            for i, val in enumerate(rows):
                rect = Rectangle(width=col_width, height=row_h,
                                  fill_color=color,
                                  fill_opacity=0.12 + 0.06*i,
                                  stroke_color=color, stroke_width=1.2)
                txt = Text(val, font_size=16, color=color)
                txt.move_to(rect)
                cell = VGroup(rect, txt)
                cell.shift(DOWN * i * row_h)
                cells.add(cell)
            head = Text(header, font_size=18, color=color, weight=BOLD)
            head.next_to(cells, UP, buff=0.18)
            return VGroup(head, cells)

        def section_title(txt, color=WHITE):
            t = Text(txt, font_size=26, color=color, weight=BOLD)
            t.to_edge(UP, buff=0.35)
            return t

        def underline(mob, color):
            line = Line(mob.get_left(), mob.get_right(),
                        color=color, stroke_width=2)
            line.next_to(mob, DOWN, buff=0.06)
            return line

        # ── SCENE 1: Title ────────────────────────────────────────────────
        title = Text("Self-Attention in Transformers",
                     font_size=38, color=WHITE, weight=BOLD)
        sub   = Text("How tokens attend to each other",
                     font_size=20, color=MUTED)
        sub.next_to(title, DOWN, buff=0.25)
        brand = VGroup(title, sub).move_to(ORIGIN)

        self.play(FadeIn(title, shift=UP*0.4), run_time=1.0)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(brand), run_time=0.5)

        # ── SCENE 2: Input tokens ─────────────────────────────────────────
        hdr2 = section_title("Step 1 — Input Tokens", C_TOKEN)
        ul2  = underline(hdr2, C_TOKEN)

        tok_boxes = VGroup(*[
            make_box(tok, C_TOKEN, width=1.2, height=0.58, font_size=24)
            for tok in tokens
        ]).arrange(RIGHT, buff=0.55).move_to(ORIGIN + DOWN*0.3)

        idx_labels = VGroup(*[
            Text(f"x{i}", font_size=15, color=MUTED).next_to(tok_boxes[i], DOWN, buff=0.18)
            for i in range(n)
        ])

        desc2 = Text("Each word is embedded into a vector xi",
                     font_size=18, color=MUTED)
        desc2.to_edge(DOWN, buff=0.5)

        self.play(Write(hdr2), Create(ul2), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, shift=UP*0.25) for b in tok_boxes],
                              lag_ratio=0.2), run_time=1.0)
        self.play(FadeIn(idx_labels), FadeIn(desc2), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(VGroup(hdr2, ul2, idx_labels, desc2)), run_time=0.4)

        # ── SCENE 3: Q/K/V Projections ────────────────────────────────────
        hdr3 = section_title("Step 2 — Query, Key & Value Projections", WHITE)
        ul3  = underline(hdr3, WHITE)
        self.play(Write(hdr3), Create(ul3), run_time=0.7)

        tok_group = tok_boxes.copy()
        self.play(tok_group.animate.scale(0.85).move_to(UP*2.2), run_time=0.7)

        q_vals = ["q1", "q2", "q3"]
        k_vals = ["k1", "k2", "k3"]
        v_vals = ["v1", "v2", "v3"]

        q_col = make_matrix_col("Q (query)", q_vals, C_QUERY)
        k_col = make_matrix_col("K (key)",   k_vals, C_KEY)
        v_col = make_matrix_col("V (value)", v_vals, C_VALUE)

        matrix_row = VGroup(q_col, k_col, v_col).arrange(RIGHT, buff=0.9)
        matrix_row.move_to(DOWN*0.55)

        arrows_q = VGroup(*[
            Arrow(tok_group[i].get_bottom(),
                  q_col[1][i].get_top(),
                  buff=0.1, stroke_width=1.5,
                  color=C_QUERY, max_tip_length_to_length_ratio=0.2)
            for i in range(n)
        ])
        arrows_k = VGroup(*[
            Arrow(tok_group[i].get_bottom(),
                  k_col[1][i].get_top(),
                  buff=0.1, stroke_width=1.5,
                  color=C_KEY, max_tip_length_to_length_ratio=0.2)
            for i in range(n)
        ])
        arrows_v = VGroup(*[
            Arrow(tok_group[i].get_bottom(),
                  v_col[1][i].get_top(),
                  buff=0.1, stroke_width=1.5,
                  color=C_VALUE, max_tip_length_to_length_ratio=0.2)
            for i in range(n)
        ])

        eq_labels = VGroup(
            Text("xi * WQ", font_size=14, color=C_QUERY).next_to(q_col, DOWN, buff=0.22),
            Text("xi * WK", font_size=14, color=C_KEY).next_to(k_col, DOWN, buff=0.22),
            Text("xi * WV", font_size=14, color=C_VALUE).next_to(v_col, DOWN, buff=0.22),
        )

        self.play(FadeIn(matrix_row), run_time=0.6)
        self.play(LaggedStart(
            AnimationGroup(*[GrowArrow(a) for a in arrows_q]),
            AnimationGroup(*[GrowArrow(a) for a in arrows_k]),
            AnimationGroup(*[GrowArrow(a) for a in arrows_v]),
            lag_ratio=0.25), run_time=1.4)
        self.play(FadeIn(eq_labels), run_time=0.5)
        self.wait(1.4)
        self.play(FadeOut(VGroup(hdr3, ul3, tok_group, arrows_q, arrows_k,
                                  arrows_v, matrix_row, eq_labels)), run_time=0.5)

        # ── SCENE 4: Attention score matrix ───────────────────────────────
        hdr4 = section_title("Step 3 — Attention Scores  (QK^T / sqrt(dk))", C_ATTN)
        ul4  = underline(hdr4, C_ATTN)
        self.play(Write(hdr4), Create(ul4), run_time=0.7)

        score_vals = [
            ["0.9", "0.2", "0.1"],
            ["0.3", "0.8", "0.4"],
            ["0.1", "0.3", "0.9"],
        ]
        cell_size = 0.78
        grid_group = VGroup()
        for r in range(3):
            for c in range(3):
                opacity = float(score_vals[r][c])
                rect = Square(side_length=cell_size,
                               fill_color=C_ATTN,
                               fill_opacity=opacity * 0.7,
                               stroke_color=C_BORDER, stroke_width=1)
                rect.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
                val = Text(score_vals[r][c], font_size=20, color=WHITE)
                val.move_to(rect)
                grid_group.add(VGroup(rect, val))

        grid_group.move_to(ORIGIN + DOWN*0.2)

        row_labels = VGroup(*[
            Text(f"q{i+1}", font_size=17, color=C_QUERY)
            .next_to(grid_group[i*3], LEFT, buff=0.22)
            for i in range(3)
        ])
        col_labels = VGroup(*[
            Text(f"k{i+1}", font_size=17, color=C_KEY)
            .next_to(grid_group[i], UP, buff=0.18)
            for i in range(3)
        ])

        formula = Text("score(qi, kj) = qi dot kj / sqrt(dk)",
                       font_size=18, color=MUTED)
        formula.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(row_labels), FadeIn(col_labels), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(cell, scale=0.7) for cell in grid_group],
                              lag_ratio=0.06), run_time=1.4)
        self.play(Write(formula), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(VGroup(hdr4, ul4, row_labels, col_labels, formula)), run_time=0.4)

        # ── SCENE 5: Softmax weights ───────────────────────────────────────
        hdr5 = section_title("Step 4 — Softmax to Attention Weights", C_ATTN)
        ul5  = underline(hdr5, C_ATTN)
        self.play(Write(hdr5), Create(ul5), run_time=0.7)

        softmax_vals = [
            ["0.70", "0.20", "0.10"],
            ["0.25", "0.60", "0.15"],
            ["0.10", "0.25", "0.65"],
        ]
        new_cells = VGroup()
        for r in range(3):
            for c in range(3):
                opacity = float(softmax_vals[r][c])
                rect = Square(side_length=cell_size,
                               fill_color=C_ATTN,
                               fill_opacity=opacity * 0.9,
                               stroke_color=C_BORDER, stroke_width=1)
                rect.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
                val = Text(softmax_vals[r][c], font_size=20, color=WHITE)
                val.move_to(rect)
                new_cells.add(VGroup(rect, val))
        new_cells.move_to(ORIGIN + DOWN*0.2)

        row_labels2 = VGroup(*[
            Text(f"q{i+1}", font_size=17, color=C_QUERY)
            .next_to(new_cells[i*3], LEFT, buff=0.22)
            for i in range(3)
        ])
        col_labels2 = VGroup(*[
            Text(f"k{i+1}", font_size=17, color=C_KEY)
            .next_to(new_cells[i], UP, buff=0.18)
            for i in range(3)
        ])
        row_sum = VGroup(*[
            Text("= 1.00", font_size=15, color=MUTED)
            .next_to(new_cells[(i+1)*3-1], RIGHT, buff=0.22)
            for i in range(3)
        ])
        softmax_note = Text("Each row sums to 1 — these are the attention weights",
                            font_size=17, color=MUTED)
        softmax_note.to_edge(DOWN, buff=0.5)

        self.play(ReplacementTransform(grid_group, new_cells), run_time=0.9)
        self.play(FadeIn(row_labels2), FadeIn(col_labels2), run_time=0.4)
        self.play(FadeIn(row_sum), Write(softmax_note), run_time=0.6)

        highlights = VGroup(*[
            SurroundingRectangle(new_cells[i*3+i],
                                  color=YELLOW, stroke_width=2.5,
                                  corner_radius=0.06, buff=0.04)
            for i in range(3)
        ])
        self.play(Create(highlights), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(VGroup(hdr5, ul5, row_labels2, col_labels2,
                                  row_sum, softmax_note, highlights)), run_time=0.4)

        # ── SCENE 6: Weighted sum of Values ───────────────────────────────
        hdr6 = section_title("Step 5 — Weighted Sum of Values to Output", C_OUTPUT)
        ul6  = underline(hdr6, C_OUTPUT)
        self.play(Write(hdr6), Create(ul6), run_time=0.7)

        v_boxes = VGroup(*[
            make_box(f"v{i+1}", C_VALUE, width=1.1, height=0.52, font_size=22)
            for i in range(3)
        ]).arrange(DOWN, buff=0.28)
        v_boxes.move_to(LEFT*3.8 + DOWN*0.2)
        v_title = Text("Value vectors", font_size=16, color=C_VALUE)
        v_title.next_to(v_boxes, UP, buff=0.2)

        w_boxes = VGroup(*[
            make_box(f"a2{i+1}={softmax_vals[1][i]}", C_ATTN,
                     width=1.45, height=0.52, font_size=18)
            for i in range(3)
        ]).arrange(DOWN, buff=0.28)
        w_boxes.move_to(ORIGIN + DOWN*0.2)
        w_title = Text("Attention weights (row 2)", font_size=16, color=C_ATTN)
        w_title.next_to(w_boxes, UP, buff=0.2)

        out_box = make_box("z2", C_OUTPUT, width=1.1, height=0.52, font_size=24)
        out_box.move_to(RIGHT*3.8 + DOWN*0.2)
        out_title = Text("Output token 2", font_size=16, color=C_OUTPUT)
        out_title.next_to(out_box, UP, buff=0.2)

        eq = Text("z2 = a21*v1 + a22*v2 + a23*v3",
                  font_size=20, color=MUTED)
        eq.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(VGroup(v_boxes, v_title)),
                  FadeIn(VGroup(w_boxes, w_title)), run_time=0.7)

        arrow_colors = [C_ATTN, ManimColor("#8E9FD4"), C_OUTPUT]
        mult_arrows = VGroup(*[
            Arrow(w_boxes[i].get_right(),
                  out_box.get_left(),
                  buff=0.1, stroke_width=1.8,
                  color=arrow_colors[i],
                  max_tip_length_to_length_ratio=0.18)
            for i in range(3)
        ])
        self.play(FadeIn(VGroup(out_box, out_title)), run_time=0.4)
        self.play(LaggedStart(*[GrowArrow(a) for a in mult_arrows],
                              lag_ratio=0.18), run_time=0.9)

        self.play(out_box.animate.scale(1.18),
                  rate_func=there_and_back, run_time=0.4)
        self.play(Write(eq), run_time=0.6)
        self.wait(1.4)
        self.play(FadeOut(VGroup(hdr6, ul6, v_boxes, v_title, w_boxes,
                                  w_title, out_box, out_title,
                                  mult_arrows, eq, new_cells)), run_time=0.6)

        # ── SCENE 7: Summary pipeline ──────────────────────────────────────
        hdr7 = section_title("Self-Attention — Full Pipeline", WHITE)
        ul7  = underline(hdr7, WHITE)
        self.play(Write(hdr7), Create(ul7), run_time=0.6)

        steps = [
            ("Input\nTokens",       C_TOKEN),
            ("Q, K, V\nProject",    WHITE),
            ("Attention\nScores",   C_ATTN),
            ("Softmax\nWeights",    C_ATTN),
            ("Weighted\nOutput",    C_OUTPUT),
        ]
        step_boxes = VGroup(*[
            make_box(label, color, width=1.45, height=0.82, font_size=17)
            for label, color in steps
        ]).arrange(RIGHT, buff=0.42).move_to(DOWN*0.2)

        arrows_pipe = VGroup(*[
            Arrow(step_boxes[i].get_right(), step_boxes[i+1].get_left(),
                  buff=0.06, stroke_width=2, color=MUTED,
                  max_tip_length_to_length_ratio=0.22)
            for i in range(len(steps)-1)
        ])

        self.play(LaggedStart(
            *[FadeIn(b, scale=0.8) for b in step_boxes],
            lag_ratio=0.15), run_time=1.2)
        self.play(LaggedStart(
            *[GrowArrow(a) for a in arrows_pipe],
            lag_ratio=0.15), run_time=0.8)

        formula_final = Text(
            "Attention(Q,K,V) = softmax( QK^T / sqrt(dk) ) * V",
            font_size=20, color=MUTED)
        formula_final.to_edge(DOWN, buff=0.5)
        self.play(Write(formula_final), run_time=0.8)
        self.wait(2.0)

        self.play(FadeOut(VGroup(hdr7, ul7, step_boxes,
                                  arrows_pipe, formula_final)), run_time=0.8)

        # ── Final card ────────────────────────────────────────────────────
        end_title = Text("Self-Attention", font_size=40, color=WHITE, weight=BOLD)
        end_sub   = Text("lets every token attend to every other token simultaneously",
                         font_size=19, color=MUTED)
        end_sub.next_to(end_title, DOWN, buff=0.3)
        VGroup(end_title, end_sub).move_to(ORIGIN)
        self.play(FadeIn(end_title, shift=UP*0.3), run_time=0.7)
        self.play(FadeIn(end_sub), run_time=0.5)
        self.wait(2.0)
        self.play(FadeOut(VGroup(end_title, end_sub)), run_time=0.7)

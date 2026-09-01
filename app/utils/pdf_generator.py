import os
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_resume_pdf(content: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    x_margin = 50
    y = height - 50
    line_height = 16
    max_chars = 95

    c.setFont("Helvetica", 11)

    for raw_line in content.splitlines():
        wrapped_lines = textwrap.wrap(raw_line, width=max_chars) or [""]
        for line in wrapped_lines:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - 50
            c.drawString(x_margin, y, line)
            y -= line_height
        y -= 4

    c.save()
    return output_path
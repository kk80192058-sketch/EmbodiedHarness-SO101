"""Create a print-at-100-percent A4 calibration kit for the SO-101 workspace."""

from pathlib import Path

from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


OUT = Path("output/pdf/so101_workspace_calibration_kit.pdf")


def label(canvas: Canvas, text: str, x: float, y: float, size: float, *, fill=black) -> None:
    canvas.setFillColor(fill)
    canvas.setFont("Helvetica-Bold", size)
    canvas.drawCentredString(x, y, text)


def draw_l_corner(canvas: Canvas, x: float, y: float, sx: int, sy: int, length: float = 25 * mm) -> None:
    """Draw a thick L with its corner at (x, y), directed by sx/sy."""
    canvas.setStrokeColor(black)
    canvas.setLineWidth(5 * mm)
    canvas.line(x, y, x + sx * length, y)
    canvas.line(x, y, x, y + sy * length)


def draw_reference_board(canvas: Canvas) -> None:
    width, height = landscape(A4)
    margin = 10 * mm
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    label(canvas, "SO-101 WORKSPACE REFERENCE BOARD", width / 2, height - 12 * mm, 13)
    label(canvas, "Print at 100% - landscape A4 - do not scale", width / 2, height - 19 * mm, 8)

    left, bottom = margin, margin
    right, top = width - margin, height - 28 * mm
    canvas.setStrokeColor(black)
    canvas.setLineWidth(1.2 * mm)
    canvas.rect(left, bottom, right - left, top - bottom, stroke=1, fill=0)
    draw_l_corner(canvas, left, top, 1, -1)
    draw_l_corner(canvas, right, top, -1, -1)
    draw_l_corner(canvas, right, bottom, -1, 1)
    draw_l_corner(canvas, left, bottom, 1, 1)
    label(canvas, "TL  (0, 0 mm)", left + 32 * mm, top - 6 * mm, 10)
    label(canvas, "TR  (277, 0 mm)", right - 32 * mm, top - 6 * mm, 10)
    label(canvas, "BL  (0, 172 mm)", left + 35 * mm, bottom + 7 * mm, 10)
    label(canvas, "BR  (277, 172 mm)", right - 35 * mm, bottom + 7 * mm, 10)
    label(canvas, "Place flat in view. All four L corners must be visible.", width / 2, top - 10 * mm, 8)

    # Print-scale checks, useful even when printer settings are unknown.
    canvas.setLineWidth(0.6 * mm)
    check_y = 24 * mm
    check_x = width / 2 - 50 * mm
    canvas.line(check_x, check_y, check_x + 100 * mm, check_y)
    for tick in (0, 50, 100):
        canvas.line(check_x + tick * mm, check_y - 3 * mm, check_x + tick * mm, check_y + 3 * mm)
    label(canvas, "100 mm print-scale check", width / 2, check_y + 5 * mm, 7)
    label(canvas, "Measured line must be exactly 100 mm", width / 2, check_y - 7 * mm, 7)
    canvas.showPage()


TAG_PATTERNS = {
    "TL": ["10101", "01010", "11100", "00111", "11001"],
    "TR": ["11010", "00101", "10111", "01001", "11100"],
    "BR": ["01110", "11001", "00111", "10100", "01011"],
    "BL": ["10011", "01101", "11010", "00110", "11101"],
}


def draw_tag(canvas: Canvas, name: str, x: float, y: float, size: float = 80 * mm) -> None:
    """A 80 mm cut-out marker: black border + distinct 5x5 binary interior."""
    canvas.setStrokeColor(black)
    canvas.setLineWidth(0.35 * mm)
    canvas.setDash(2 * mm, 1.5 * mm)
    canvas.rect(x, y, size, size, stroke=1, fill=0)
    canvas.setDash()
    border = 5 * mm
    canvas.setFillColor(black)
    canvas.rect(x + 2 * mm, y + 2 * mm, size - 4 * mm, size - 4 * mm, stroke=0, fill=1)
    inner_x, inner_y = x + border, y + border
    inner = size - 2 * border
    canvas.setFillColor(white)
    canvas.rect(inner_x, inner_y, inner, inner, stroke=0, fill=1)
    cells = 5
    cell = inner / cells
    for row, bits in enumerate(TAG_PATTERNS[name]):
        for col, bit in enumerate(bits):
            if bit == "1":
                canvas.setFillColor(black)
                canvas.rect(inner_x + col * cell, inner_y + (cells - 1 - row) * cell, cell, cell, stroke=0, fill=1)
    # White label plate over the center makes manual correspondence unambiguous.
    plate_w, plate_h = 33 * mm, 12 * mm
    canvas.setFillColor(white)
    canvas.rect(x + (size - plate_w) / 2, y + (size - plate_h) / 2, plate_w, plate_h, stroke=0, fill=1)
    label(canvas, name, x + size / 2, y + size / 2 - 2 * mm, 14)
    label(canvas, "80 x 80 mm", x + size / 2, y - 5 * mm, 7)


def draw_corner_tags(canvas: Canvas) -> None:
    width, height = A4
    canvas.setFillColor(white)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    label(canvas, "SO-101 WORKSPACE CORNER TAGS", width / 2, height - 13 * mm, 13)
    label(canvas, "Print at 100% - cut on dashed lines - place around the full task workspace", width / 2, height - 20 * mm, 8)
    draw_tag(canvas, "TL", 15 * mm, 160 * mm)
    draw_tag(canvas, "TR", 115 * mm, 160 * mm)
    draw_tag(canvas, "BL", 15 * mm, 50 * mm)
    draw_tag(canvas, "BR", 115 * mm, 50 * mm)
    label(canvas, "Keep every tag flat and fully visible. Do not place objects on them.", width / 2, 18 * mm, 8)
    canvas.showPage()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(OUT), pagesize=landscape(A4))
    draw_reference_board(canvas)
    canvas.setPageSize(A4)
    draw_corner_tags(canvas)
    canvas.save()
    print(OUT)


if __name__ == "__main__":
    main()

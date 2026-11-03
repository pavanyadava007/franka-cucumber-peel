"""Build a labelled JPEG contact sheet.
Usage: python -I contact_sheet.py <out.jpg> <cols> <title> label1=path1 [label2=path2 ...]
Each tile is scaled to height 240 and padded to the widest tile in the grid."""
import sys
from PIL import Image, ImageDraw, ImageFont

out, cols, title = sys.argv[1], int(sys.argv[2]), sys.argv[3]
items = [a.rsplit("=", 1) for a in sys.argv[4:]]
H = 240
tiles = []
for label, path in items:
    im = Image.open(path).convert("RGB")
    w = int(im.width * H / im.height)
    tiles.append((label, im.resize((w, H))))
W = max(t[1].width for t in tiles)
cap = 22
rows = (len(tiles) + cols - 1) // cols
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception:
    font = tfont = ImageFont.load_default()
sheet = Image.new("RGB", (cols * W, 34 + rows * (H + cap)), (255, 255, 255))
d = ImageDraw.Draw(sheet)
d.text((8, 8), title, fill=(0, 0, 0), font=tfont)
for i, (label, im) in enumerate(tiles):
    r, c = divmod(i, cols)
    x, y = c * W, 34 + r * (H + cap)
    sheet.paste(im, (x + (W - im.width) // 2, y + cap))
    d.text((x + 4, y + 3), label, fill=(0, 0, 0), font=font)
sheet.save(out, quality=88)
print("wrote", out, sheet.size)

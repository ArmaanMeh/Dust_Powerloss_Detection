import re
from pathlib import Path
 
import cv2 
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:5000"
OUTPUT = ROOT / "Solar_Dust_Classification_Demo.mp4" 
WIDTH, HEIGHT, FPS = 1280, 720, 24
FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_REGULAR = str(FONT_DIR / "segoeui.ttf")
FONT_BOLD = str(FONT_DIR / "segoeuib.ttf") 
COLORS = {
    "bg": "#F3F7F5",
    "white": "#FFFFFF",
    "ink": "#172B36", 
    "muted": "#64777B", 
    "green": "#147D59", 
    "green_light": "#E8F5EE",
    "line": "#E2EBE5",
    "amber": "#C87B1A", 
    "amber_light": "#FFF3E1",
    "dark": "#102A24",
}


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def classify(path):
    with path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (path.name, image_file, "image/jpeg")},
            timeout=90,
        )
    response.raise_for_status()
    label_match = re.search(r"<h1>(Clean|Dusty) panel</h1>", response.text)
    score_match = re.search(r"<strong>([0-9.]+)%</strong>", response.text)
    if not label_match or not score_match:
        raise RuntimeError(f"Unexpected classification page returned for {path.name}") 
    return {
        "file": path,
        "label": label_match.group(1),
        "score": float(score_match.group(1)),
        "status": response.status_code, 
    }


def centered(draw, text, y, text_font, fill, width=WIDTH): 
    bounds = draw.textbbox((0, 0), text, font=text_font) 
    draw.text(((width - (bounds[2] - bounds[0])) / 2, y), text, font=text_font, fill=fill)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_image(path):
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (490, 310), method=Image.Resampling.LANCZOS)
 

def frame(kind, item=None, progress=0.0):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["bg"]) 
    draw = ImageDraw.Draw(canvas)

    if kind == "intro":
        canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["dark"])
        draw = ImageDraw.Draw(canvas)
        rounded(draw, (80, 78, 1200, 642), 30, "#173A31", outline="#2D5849", width=2)
        rounded(draw, (128, 135, 185, 192), 17, COLORS["green"])
        draw.text((140, 137), "☼", font=font(36, True), fill=COLORS["white"])
        draw.text((128, 252), "SOLAR PANEL INSPECTION", font=font(17, True), fill="#A7D9BF")
        draw.text((128, 298), "Solar Dust Detector", font=font(57, True), fill=COLORS["white"])
        draw.text((130, 390), "Live image classification with MobileNet + Flask", font=font(24), fill="#C3D6CC") 
        rounded(draw, (130, 478, 432, 530), 14, "#245342")
        draw.text((151, 492), "LOCAL DEMO  ·  127.0.0.1:5000", font=font(16, True), fill="#D8F0E2") 
        centered(draw, "Clean vs dusty  •  Real images from the included dataset", 604, font(15), "#B1C7BD")
        return canvas

    draw.rectangle((0, 0, WIDTH, 72), fill=COLORS["white"])
    draw.line((0, 71, WIDTH, 71), fill=COLORS["line"], width=2)
    rounded(draw, (58, 16, 98, 56), 12, COLORS["green"]) 
    draw.text((68, 18), "☼", font=font(27, True), fill=COLORS["white"])
    draw.text((112, 22), "Solar Dust Detector", font=font(19, True), fill=COLORS["ink"])
    rounded(draw, (958, 21, 1224, 51), 15, COLORS["green_light"])
    draw.ellipse((975, 31, 984, 40), fill=COLORS["green"])
    draw.text((993, 26), "API READY  ·  FLASK", font=font(13, True), fill=COLORS["green"])

    if kind == "home":
        draw.text((74, 121), "SOLAR PANEL INSPECTION", font=font(13, True), fill=COLORS["green"])
        draw.text((74, 153), "See the dust. Protect the power.", font=font(39, True), fill=COLORS["ink"]) 
        draw.text((76, 210), "Upload a panel image to classify it as clean or dusty.", font=font(18), fill=COLORS["muted"])
        rounded(draw, (74, 279, 1206, 645), 22, COLORS["white"], COLORS["line"], 2)
        draw.text((117, 318), "Classify a panel", font=font(26, True), fill=COLORS["ink"]) 
        draw.text((117, 357), "Choose an image from the repository dataset.", font=font(16), fill=COLORS["muted"])
        rounded(draw, (117, 410, 1163, 548), 17, "#F8FBF9", "#A9CBB8", 2) 
        centered(draw, "↑", 423, font(34, True), COLORS["green"])
        centered(draw, "Choose an image from your device", 468, font(19, True), COLORS["ink"])
        centered(draw, "JPG  ·  JPEG  ·  PNG", 502, font(13), COLORS["muted"])
        rounded(draw, (117, 573, 1163, 621), 12, COLORS["green"])
        centered(draw, "Analyze image   →", 584, font(17, True), COLORS["white"])
        rounded(draw, (78, 91, 416, 118), 12, COLORS["white"], COLORS["line"])
        draw.text((92, 95), "http://127.0.0.1:5000/", font=font(13), fill=COLORS["muted"])
        return canvas

    if kind == "summary":
        draw.text((74, 118), "DEMO COMPLETE", font=font(13, True), fill=COLORS["green"])
        draw.text((74, 153), "Two images. Two live predictions.", font=font(38, True), fill=COLORS["ink"])
        draw.text((76, 208), "Each result below came from an HTTP POST to the running Flask API.", font=font(17), fill=COLORS["muted"])
        for x, result in ((74, CLEAN), (654, DUSTY)):
            rounded(draw, (x, 278, x + 552, 535), 22, COLORS["white"], COLORS["line"], 2)
            draw.text((x + 30, 307), "CLEAN SAMPLE" if result["label"] == "Clean" else "DUSTY SAMPLE", font=font(13, True), fill=COLORS["green"] if result["label"] == "Clean" else COLORS["amber"])
            draw.text((x + 30, 345), result["label"], font=font(40, True), fill=COLORS["ink"])
            draw.text((x + 30, 407), f"Dust score   {result['score']:.1f}%", font=font(19, True), fill=COLORS["green"] if result["label"] == "Clean" else COLORS["amber"])
            draw.text((x + 30, 456), f"POST /predict    HTTP {result['status']}", font=font(14), fill=COLORS["muted"])
        rounded(draw, (74, 579, 1206, 636), 14, COLORS["dark"])
        centered(draw, "MobileNet model  •  224 × 224 preprocessing  •  Local Flask API", 596, font(17, True), COLORS["white"]) 
        return canvas 

    rounded(draw, (74, 112, 1206, 657), 22, COLORS["white"], COLORS["line"], 2)
    draw.text((112, 143), "IMAGE CLASSIFICATION", font=font(13, True), fill=COLORS["green"])
    draw.text((112, 173), item["file"].name, font=font(20, True), fill=COLORS["ink"])
    canvas.paste(make_image(item["file"]), (112, 222))
    draw = ImageDraw.Draw(canvas)
    rounded(draw, (112, 550, 602, 617), 12, "#F8FBF9", COLORS["line"]) 
    draw.text((132, 560), "DATASET SAMPLE", font=font(12, True), fill=COLORS["green"])
    draw.text((132, 580), "Uploaded image · RGB · resized to 224 × 224", font=font(13), fill=COLORS["muted"]) 
    draw.text((650, 239), "Flask inference endpoint", font=font(20, True), fill=COLORS["ink"])
    rounded(draw, (650, 284, 1155, 347), 12, "#F7FAF8", COLORS["line"])
    draw.text((671, 296), "POST", font=font(15, True), fill=COLORS["green"])
    draw.text((736, 296), "/predict", font=font(18, True), fill=COLORS["ink"])
    draw.text((671, 321), "multipart/form-data  ·  file", font=font(13), fill=COLORS["muted"])
 
    if kind == "ready":
        draw.text((650, 382), "Image selected", font=font(20, True), fill=COLORS["ink"])
        draw.text((650, 416), "Ready to send to the Flask prediction API.", font=font(15), fill=COLORS["muted"]) 
        rounded(draw, (650, 466, 1155, 481), 7, "#E5EEE8")
        draw.text((650, 500), "NEXT  →  POST /predict", font=font(15, True), fill=COLORS["green"]) 
        rounded(draw, (650, 551, 1155, 613), 12, "#F7FAF8", COLORS["line"]) 
        draw.text((671, 562), "MODEL", font=font(12, True), fill=COLORS["muted"]) 
        draw.text((671, 582), "MobileNet · binary classifier", font=font(15, True), fill=COLORS["ink"])
    elif kind == "processing":
        draw.text((650, 382), "Running MobileNet inference", font=font(20, True), fill=COLORS["ink"])
        draw.text((650, 416), "Preprocess  →  predict  →  render result", font=font(15), fill=COLORS["muted"]) 
        rounded(draw, (650, 466, 1155, 481), 7, "#E5EEE8")
        progress_width = max(18, int(505 * progress))
        rounded(draw, (650, 466, 650 + progress_width, 481), 7, COLORS["green"])
        draw.text((650, 500), f"Analyzing image  ·  {int(progress * 100)}%", font=font(15, True), fill=COLORS["green"])
        rounded(draw, (650, 551, 1155, 613), 12, "#F7FAF8", COLORS["line"])
        draw.text((671, 562), "MODEL", font=font(12, True), fill=COLORS["muted"])
        draw.text((671, 582), "MobileNet · binary classifier", font=font(15, True), fill=COLORS["ink"]) 
    else:
        is_dusty = item["label"] == "Dusty"
        accent = COLORS["amber"] if is_dusty else COLORS["green"]
        tint = COLORS["amber_light"] if is_dusty else COLORS["green_light"]
        draw.text((650, 379), "CLASSIFICATION RESULT", font=font(13, True), fill=accent) 
        draw.text((650, 407), f"{item['label']} panel", font=font(37, True), fill=COLORS["ink"])
        rounded(draw, (650, 468, 1155, 531), 12, tint) 
        draw.text((671, 485), "Dust score", font=font(15), fill=COLORS["muted"])
        draw.text((1035, 480), f"{item['score']:.1f}%", font=font(22, True), fill=accent) 
        rounded(draw, (650, 551, 1155, 613), 12, "#F7FAF8", COLORS["line"])
        draw.ellipse((672, 576, 681, 585), fill=COLORS["green"])
        draw.text((696, 565), f"HTTP {item['status']}  ·  Flask response rendered", font=font(14, True), fill=COLORS["ink"])
    return canvas


def main():
    home = requests.get(f"{BASE_URL}/", timeout=10)
    home.raise_for_status()
    global CLEAN, DUSTY 
    clean_path = ROOT / "Detect_Solar_dust" / "clean-20260925T100043Z-1-001" / "clean" / "20210917_151202.jpg"
    dusty_path = ROOT / "Detect_Solar_dust" / "dirty-20260925T100044Z-1-001" / "dirty" / "20210916_094041.jpg"
    CLEAN = classify(clean_path) 
    DUSTY = classify(dusty_path)
 
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT), 
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 video writer for {OUTPUT}")

    scenes = [
        ("intro", None, 3),
        ("home", None, 5), 
        ("ready", CLEAN, 2), 
        ("processing", CLEAN, 8),
        ("result", CLEAN, 5), 
        ("ready", DUSTY, 2),
        ("processing", DUSTY, 8), 
        ("result", DUSTY, 5),
        ("summary", None, 2),
    ]
    try: 
        for kind, item, duration in scenes: 
            frame_count = duration * FPS
            for index in range(frame_count):
                progress = (index + 1) / frame_count if kind == "processing" else 0
                image = frame(kind, item, progress)
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Video: {OUTPUT}")
    print(f"Duration: {sum(scene[2] for scene in scenes)} seconds ({sum(scene[2] for scene in scenes) * FPS} frames)")
    print(f"Clean API prediction: {CLEAN['label']} ({CLEAN['score']:.1f}% dust)")
    print(f"Dusty API prediction: {DUSTY['label']} ({DUSTY['score']:.1f}% dust)")
    print(f"File size: {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

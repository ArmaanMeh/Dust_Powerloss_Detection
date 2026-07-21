from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Detect_Solar_dust" / "Kaggle_PV_Panel_Defect_Dataset"
OUTPUT = ROOT / "Solar_Dust_Kaggle_Classification_Demo.mp4" 
BASE_URL = "http://127.0.0.1:5000"
WIDTH, HEIGHT, FPS = 1280, 720, 24
REGULAR = r"C:\Windows\Fonts\segoeui.ttf"
BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
COLORS = {
    "background": "#F3F7F5",
    "white": "#FFFFFF",
    "ink": "#172B36",
    "muted": "#64777B",
    "green": "#147D59",
    "green_light": "#E8F5EE",
    "amber": "#C87B1A",
    "amber_light": "#FFF3E1",
    "line": "#E2EBE5", 
    "dark": "#102A24",
}
ATTRIBUTION = "Kaggle: PV Panel Defect Dataset  ·  Alicja Lenarczyk  ·  CC BY-NC-SA 4.0"

 
def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else REGULAR, size) 


def centered(draw, text, y, text_font, fill, width=WIDTH):
    bounds = draw.textbbox((0, 0), text, font=text_font)
    draw.text(((width - bounds[2] + bounds[0]) / 2, y), text, font=text_font, fill=fill)

 
def round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

 
def classify_image(path):
    with path.open("rb") as uploaded:
        response = requests.post(
            f"{BASE_URL}/predict", 
            files={"file": (path.name, uploaded, "image/jpeg")},
            timeout=90, 
        ) 
    response.raise_for_status() 
    label = re.search(r"<h1>(Clean|Dusty) panel</h1>", response.text)
    score = re.search(r"<strong>([0-9.]+)%</strong>", response.text)
    if label is None or score is None:
        raise RuntimeError(f"Unexpected response from the Flask API for {path.name}")
    return {
        "path": path, 
        "category": path.parent.name,
        "prediction": label.group(1),
        "score": float(score.group(1)),
        "http_status": response.status_code,
    }


def make_header(draw, page_label, top=0): 
    draw.rectangle((0, top, WIDTH, top + 74), fill=COLORS["white"])
    draw.line((0, top + 73, WIDTH, top + 73), fill=COLORS["line"], width=2)
    round_rect(draw, (56, top + 16, 97, top + 57), 12, COLORS["green"])
    draw.text((65, top + 17), "☼", font=font(28, True), fill=COLORS["white"])
    draw.text((112, top + 21), "Solar Dust Detector", font=font(19, True), fill=COLORS["ink"]) 
    round_rect(draw, (1002, top + 21, 1223, top + 51), 15, COLORS["green_light"])
    draw.ellipse((1017, top + 31, 1026, top + 40), fill=COLORS["green"])
    draw.text((1036, top + 26), page_label, font=font(12, True), fill=COLORS["green"]) 
 
 
def make_intro():
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["dark"])
    draw = ImageDraw.Draw(canvas)
    round_rect(draw, (75, 68, 1205, 651), 30, "#173A31", outline="#2D5849", width=2)
    round_rect(draw, (124, 117, 183, 176), 17, COLORS["green"])
    draw.text((137, 118), "☼", font=font(38, True), fill=COLORS["white"]) 
    draw.text((124, 218), "LIVE FLASK API · KAGGLE TEST IMAGES", font=font(17, True), fill="#A7D9BF")
    draw.text((124, 268), "Dust & bird-drop", font=font(56, True), fill=COLORS["white"])
    draw.text((124, 334), "classification demo", font=font(56, True), fill=COLORS["white"])
    draw.text((128, 433), "11 new solar-panel images  ·  2 Kaggle categories", font=font(22), fill="#C3D6CC")
    round_rect(draw, (128, 500, 530, 555), 14, "#245342")
    draw.text((148, 516), "REAL PREDICTIONS  ·  /predict", font=font(16, True), fill="#D8F0E2")
    centered(draw, ATTRIBUTION, 608, font(14), "#B1C7BD")
    return canvas


def make_overview(bird_count, dusty_count):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    make_header(draw, "DEMO OVERVIEW")
    draw.text((75, 123), "LIVE WALKTHROUGH · ONE IMAGE AT A TIME", font=font(13, True), fill=COLORS["green"])
    draw.text((75, 158), "Watch each prediction", font=font(40, True), fill=COLORS["ink"])
    draw.text((77, 218), "Select a sample, send it to Flask, wait for MobileNet, and see the result.", font=font(18), fill=COLORS["muted"])

    for x, category, count, color, tint in ( 
        (75, "Bird-drop", bird_count, COLORS["amber"], COLORS["amber_light"]),
        (660, "Dusty", dusty_count, COLORS["green"], COLORS["green_light"]),
    ):
        round_rect(draw, (x, 291, x + 545, 514), 22, COLORS["white"], COLORS["line"], 2)
        round_rect(draw, (x + 30, 320, x + 220, 358), 17, tint) 
        draw.text((x + 46, 330), category.upper(), font=font(15, True), fill=color)
        draw.text((x + 30, 381), f"{count} images", font=font(31, True), fill=COLORS["ink"]) 
        draw.text((x + 30, 434), "Each upload gets its own live API response.", font=font(15), fill=COLORS["muted"])

    round_rect(draw, (75, 546, 1205, 627), 17, COLORS["dark"])
    draw.text((103, 561), "Model:", font=font(16, True), fill="#A7D9BF")
    draw.text((172, 561), "Binary classification (Clean / Dusty). Bird-drop is a source label, not an output class.", font=font(16), fill=COLORS["white"])
    draw.text((103, 592), ATTRIBUTION, font=font(13), fill="#B1C7BD")
    return canvas 

 
def make_browser_scene(item, index, total, stage, progress=0.0):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"]) 
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 41), fill="#E9EEF2")
    for x, color in ((19, "#E68077"), (39, "#E5B85B"), (59, "#67B987")): 
        draw.ellipse((x, 13, x + 11, 24), fill=color)
    round_rect(draw, (90, 6, 624, 35), 13, COLORS["white"])
    draw.text((108, 11), f"Solar Dust Detector  ·  {item['path'].name}", font=font(13), fill=COLORS["muted"]) 
    round_rect(draw, (640, 6, 1209, 35), 13, "#F6F8F9")
    draw.text((657, 11), "127.0.0.1:5000/", font=font(13), fill=COLORS["muted"])
    make_header(draw, "CLASSIFICATION DEMO", top=41)
    draw.text((68, 126), f"IMAGE {index:02d} OF {total:02d}   ·   {item['category'].upper()} DATASET SAMPLE", font=font(13, True), fill=COLORS["green"])
    draw.text((68, 157), "Solar panel image classifier", font=font(29, True), fill=COLORS["ink"])
    draw.text((70, 197), "Inspect the image and follow its live upload to the Flask API.", font=font(15), fill=COLORS["muted"])
 
    image = item["preview"] 
    canvas.paste(image, (68, 239))
    draw = ImageDraw.Draw(canvas)
    round_rect(draw, (68, 595, 617, 644), 12, COLORS["white"], COLORS["line"])
    draw.text((86, 603), "SOURCE CLASS", font=font(11, True), fill=COLORS["muted"])
    draw.text((86, 620), f"{item['category']}   ·   {item['path'].name}", font=font(13, True), fill=COLORS["ink"])

    panel_x = 648 
    round_rect(draw, (panel_x, 143, 1212, 644), 20, COLORS["white"], COLORS["line"], 2)
    draw.text((panel_x + 29, 165), "Classify a panel", font=font(23, True), fill=COLORS["ink"])
    draw.text((panel_x + 30, 199), "Upload one image to check its surface condition.", font=font(14), fill=COLORS["muted"])
 
    round_rect(draw, (panel_x + 29, 235, 1181, 365), 14, "#F8FBF9", "#A9CBB8" if stage == "selected" else COLORS["line"], 2)
    if stage == "selected":
        draw.ellipse((panel_x + 53, 267, panel_x + 89, 303), fill=COLORS["green_light"])
        draw.text((panel_x + 63, 269), "✓", font=font(24, True), fill=COLORS["green"])
        draw.text((panel_x + 103, 259), "Image selected", font=font(16, True), fill=COLORS["ink"])
        draw.text((panel_x + 103, 286), item["path"].name, font=font(13), fill=COLORS["muted"])
        draw.text((panel_x + 53, 328), "JPG image ready to upload", font=font(12), fill=COLORS["muted"])
    elif stage == "upload":
        draw.text((panel_x + 52, 256), "Uploading selected image…", font=font(17, True), fill=COLORS["ink"])
        draw.text((panel_x + 53, 286), item["path"].name, font=font(13), fill=COLORS["muted"])
        round_rect(draw, (panel_x + 53, 324, panel_x + 533, 337), 6, "#E5EEE8") 
        round_rect(draw, (panel_x + 53, 324, panel_x + 53 + int(480 * progress), 337), 6, COLORS["green"]) 
    elif stage == "processing":
        draw.ellipse((panel_x + 56, 269, panel_x + 94, 307), outline="#B6D9C5", width=5)
        start_angle = int((progress * 360) % 360)
        draw.arc((panel_x + 56, 269, panel_x + 94, 307), start=start_angle, end=start_angle + 230, fill=COLORS["green"], width=5)
        draw.text((panel_x + 108, 261), "Running MobileNet inference…", font=font(16, True), fill=COLORS["ink"])
        draw.text((panel_x + 108, 289), "Waiting for the Flask prediction response", font=font(13), fill=COLORS["muted"])
        draw.text((panel_x + 52, 329), "POST /predict   ·   multipart/form-data", font=font(12, True), fill=COLORS["green"]) 
    else: 
        is_dusty = item["prediction"] == "Dusty"
        result_color = COLORS["amber"] if is_dusty else COLORS["green"]
        result_tint = COLORS["amber_light"] if is_dusty else COLORS["green_light"]
        draw.ellipse((panel_x + 53, 268, panel_x + 89, 304), fill=result_tint)
        draw.text((panel_x + 64, 270), "✓", font=font(24, True), fill=result_color)
        draw.text((panel_x + 103, 258), "Prediction complete", font=font(16, True), fill=COLORS["ink"]) 
        draw.text((panel_x + 103, 286), f"HTTP {item['http_status']}  ·  {item['path'].name}", font=font(13), fill=COLORS["muted"])

    round_rect(draw, (panel_x + 29, 384, 1181, 438), 12, COLORS["green"])
    button_text = {
        "selected": "Analyze image   →",
        "upload": "Uploading image…",
        "processing": "Analyzing image with MobileNet…", 
        "result": f"Result: {item.get('prediction', 'Clean or Dusty')}",
    }[stage] 
    centered(draw, button_text, 398, font(16, True), COLORS["white"], width=panel_x + 1212)

    if stage == "result":
        dusty = item["prediction"] == "Dusty"
        accent = COLORS["amber"] if dusty else COLORS["green"] 
        tint = COLORS["amber_light"] if dusty else COLORS["green_light"]
        round_rect(draw, (panel_x + 29, 456, 1181, 552), 13, tint)
        draw.text((panel_x + 49, 468), "MODEL CLASSIFICATION", font=font(11, True), fill=accent)
        draw.text((panel_x + 49, 488), f"{item['prediction']} panel", font=font(25, True), fill=COLORS["ink"])
        draw.text((panel_x + 428, 479), "Dust score", font=font(13), fill=COLORS["muted"])
        draw.text((panel_x + 464, 500), f"{item['score']:.1f}%", font=font(18, True), fill=accent)
        draw.text((panel_x + 31, 573), "This model predicts Clean or Dusty, not bird-dropping as a separate class.", font=font(11), fill=COLORS["muted"])
    else:
        draw.text((panel_x + 31, 471), "MODEL", font=font(11, True), fill=COLORS["muted"])
        draw.text((panel_x + 31, 491), "MobileNet · binary Clean / Dusty classifier", font=font(14, True), fill=COLORS["ink"]) 
        draw.text((panel_x + 31, 526), "API endpoint", font=font(11, True), fill=COLORS["muted"])
        draw.text((panel_x + 31, 545), "POST  /predict", font=font(14, True), fill=COLORS["green"])

    draw.text((70, 675), ATTRIBUTION, font=font(11), fill=COLORS["muted"])
    return canvas
 
 
def make_outro(items):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["dark"])
    draw = ImageDraw.Draw(canvas)
    round_rect(draw, (75, 68, 1205, 651), 30, "#173A31", outline="#2D5849", width=2)
    draw.text((127, 120), "DEMO COMPLETE", font=font(15, True), fill="#A7D9BF")
    draw.text((127, 164), "Every new sample reached Flask.", font=font(38, True), fill=COLORS["white"])
    draw.text((130, 232), f"{len(items)} live HTTP 200 predictions from the Kaggle test split.", font=font(20), fill="#C3D6CC")
    bird = [item for item in items if item["category"] == "Bird-drop"]
    dusty = [item for item in items if item["category"] == "Dusty"]
    summary = [
        ("Bird-drop samples", sum(item["prediction"] == "Dusty" for item in bird), len(bird)),
        ("Dusty samples", sum(item["prediction"] == "Dusty" for item in dusty), len(dusty)),
    ]
    for row, (label, predictions, count) in enumerate(summary): 
        y = 315 + row * 79
        round_rect(draw, (129, y, 1150, y + 59), 15, "#245342")
        draw.text((153, y + 16), label, font=font(19, True), fill=COLORS["white"])
        draw.text((876, y + 16), f"{predictions} / {count} predicted Dusty", font=font(17, True), fill="#D8F0E2")
    draw.text((130, 493), "Bird-drop is a source label only; the existing model is binary, not a bird-dropping detector.", font=font(16, True), fill="#F5D79F")
    draw.text((130, 548), ATTRIBUTION, font=font(14), fill="#B1C7BD")
    draw.text((130, 585), "Non-commercial research/demo subset · source and license notice stored with the images.", font=font(13), fill="#B1C7BD")
    return canvas 
 

def main():
    health = requests.get(f"{BASE_URL}/", timeout=10)
    health.raise_for_status() 
    paths = []
    for category in ("Bird-drop", "Dusty"):
        folder = DATASET / category
        image_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not image_paths:
            raise RuntimeError(f"No image samples found in {folder}")
        paths.extend(image_paths)

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened(): 
        raise RuntimeError(f"Could not create MP4 output at {OUTPUT}")

    expected_seconds = 4 + 5 + len(paths) * (2 + 1 + 3 + 3) + 6
    frame_total = expected_seconds * FPS
    written = 0

    def write_scene(scene, duration):
        nonlocal written
        base_frame = cv2.cvtColor(np.asarray(scene), cv2.COLOR_RGB2BGR)
        for _ in range(round(duration * FPS)): 
            video_frame = base_frame.copy()
            cv2.rectangle(video_frame, (0, HEIGHT - 5), (WIDTH, HEIGHT), (230, 238, 232), -1)
            bar_width = int(WIDTH * (written + 1) / frame_total)
            cv2.rectangle(video_frame, (0, HEIGHT - 5), (bar_width, HEIGHT), (89, 125, 20), -1)
            writer.write(video_frame)
            written += 1

    items = []
    try: 
        write_scene(make_intro(), 4) 
        write_scene(make_overview( 
            sum(path.parent.name == "Bird-drop" for path in paths),
            sum(path.parent.name == "Dusty" for path in paths),
        ), 5)

        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, path in enumerate(paths, 1):
                preview = Image.open(path).convert("RGB")
                preview = ImageOps.fit(preview, (549, 340), method=Image.Resampling.LANCZOS)
                item = {"path": path, "category": path.parent.name, "preview": preview}

                write_scene(make_browser_scene(item, index, len(paths), "selected"), 2) 
                for upload_frame in range(FPS):
                    progress = (upload_frame + 1) / FPS
                    write_scene(make_browser_scene(item, index, len(paths), "upload", progress), 1)

                prediction = executor.submit(classify_image, path)
                process_frame = 0
                minimum_processing_frames = 3 * FPS
                while process_frame < minimum_processing_frames or not prediction.done():
                    progress = process_frame / FPS
                    write_scene(
                        make_browser_scene(item, index, len(paths), "processing", progress), 
                        1 / FPS,
                    )
                    process_frame += 1

                item.update(prediction.result())
                items.append(item)
                write_scene(make_browser_scene(item, index, len(paths), "result"), 3)

        write_scene(make_outro(items), 6)
    finally:
        writer.release() 
 
    print(f"Saved: {OUTPUT}")
    print(f"Duration: {written / FPS:.2f} seconds ({written} frames at {FPS} fps)")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"File size: {OUTPUT.stat().st_size} bytes")
    print(f"Images sent through Flask /predict: {len(items)}")
    for item in items:
        print(f"{item['category']} | {item['path'].name} | {item['prediction']} | dust score {item['score']:.1f}%") 


if __name__ == "__main__":
    main()

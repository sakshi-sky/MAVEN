import cv2
import os

VIDEO_ROOT = "data/raw/IEMOCAP_full_release"
FRAME_ROOT = "data/frames"

os.makedirs(FRAME_ROOT, exist_ok=True)

def extract(video_path, save_dir, fps=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps // fps)

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            cv2.imwrite(
                os.path.join(save_dir, f"{saved}.jpg"),
                frame
            )
            saved += 1

        frame_id += 1

    cap.release()

# TODO: loop over all sessions and videos

if __name__ == "__main__":
    print("Starting frame extraction...")

    for session in os.listdir(VIDEO_ROOT):
        if not session.startswith("Session"):
            continue

        session_path = os.path.join(VIDEO_ROOT, session)

        for root, dirs, files in os.walk(session_path):
            for file in files:
                if file.endswith(".avi"):
                    video_path = os.path.join(root, file)

                    save_dir = os.path.join(FRAME_ROOT, session, file.replace(".avi", ""))
                    os.makedirs(save_dir, exist_ok=True)

                    print(f"Extracting: {video_path}")
                    extract(video_path, save_dir)

    print("Done.")

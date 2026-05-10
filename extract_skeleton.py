"""Extract MediaPipe hand landmarks for all NVGesture train+test samples within
each sample's gesture window. Saves to skeleton_landmarks.npz as dict
{sample_relative_path: (T, 21, 3) float32 array} with NaN for undetected frames.
"""
import cv2, os, re, sys, time
import numpy as np
import mediapipe as mp

VIDEO_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
OUT = '/notebooks/PMamba/dataset/Nvidia/Processed/skeleton_landmarks.npz'

mp_hands = mp.solutions.hands

def parse_annotation(path):
    samples = []
    with open(path) as f:
        for line in f:
            m_path = re.search(r'path:(\S+)', line)
            m_color = re.search(r'color:sk_color:(\d+):(\d+)', line)
            m_label = re.search(r'label:(\d+)', line)
            if m_path and m_color and m_label:
                samples.append({
                    'path': m_path.group(1),
                    'start': int(m_color.group(1)),
                    'end': int(m_color.group(2)),
                    'label': int(m_label.group(1)),
                })
    return samples

train = parse_annotation(f'{VIDEO_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test = parse_annotation(f'{VIDEO_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
all_samples = train + test
print(f'Train: {len(train)}, Test: {len(test)}, Total: {len(all_samples)}')

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.05,
    min_tracking_confidence=0.05,
    model_complexity=1,
)

results = {}
t0 = time.time()
for i, s in enumerate(all_samples):
    video = f'{VIDEO_ROOT}/{s["path"]}/sk_color.avi'
    if not os.path.exists(video):
        results[s['path']] = None
        continue
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s['start'])
    n = s['end'] - s['start']
    arr = np.full((n, 21, 3), np.nan, dtype=np.float32)
    for fi in range(n):
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            arr[fi] = np.array([[p.x, p.y, p.z] for p in lm.landmark])
    cap.release()
    detected = np.isfinite(arr[..., 0]).all(axis=-1).sum()
    results[s['path']] = arr
    if i % 50 == 0 or i == len(all_samples) - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(1, i+1) * (len(all_samples) - i - 1)
        print(f'{i+1}/{len(all_samples)} | {s["path"][-40:]} | det={detected}/{n} | elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

hands.close()
np.savez_compressed(OUT, **{k: v for k, v in results.items() if v is not None})
print(f'\nSaved {len(results)} samples to {OUT}')
print(f'Total time: {time.time()-t0:.0f}s')

"""Extract depth frames from sk_depth.avi (gesture window only) as numbered JPGs.
Output layout: /notebooks/cvpr_data/depth/{sample_dir}/000001.jpg ...
And test split: {sample_dir} n_frames label  (1-indexed, label 0..24)
"""
import cv2, os, re, sys, time
import numpy as np

VIDEO_ROOT = '/notebooks/PMamba/dataset_full/nvGesture_v1.1/nvGesture_v1'
OUT_ROOT = '/notebooks/cvpr_data/depth'
SPLITS_DIR = '/notebooks/cvpr_data/dataset_splits'
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

def parse_annotation(path):
    samples = []
    with open(path) as f:
        for line in f:
            m_path = re.search(r'path:(\S+)', line)
            m_depth = re.search(r'depth:sk_depth:(\d+):(\d+)', line)
            m_label = re.search(r'label:(\d+)', line)
            if m_path and m_depth and m_label:
                samples.append({
                    'path': m_path.group(1),
                    'start': int(m_depth.group(1)),
                    'end': int(m_depth.group(2)),
                    'label': int(m_label.group(1)) - 1,  # 1-indexed -> 0-indexed
                })
    return samples

train = parse_annotation(f'{VIDEO_ROOT}/nvgesture_train_correct_cvpr2016_v2.lst')
test = parse_annotation(f'{VIDEO_ROOT}/nvgesture_test_correct_cvpr2016_v2.lst')
print(f'Train: {len(train)}, Test: {len(test)}')

def extract(samples, split_name):
    split_lines = []
    t0 = time.time()
    for i, s in enumerate(samples):
        # sample_dir name: just the subject_dir (last part after class_XX/)
        # use full subpath like 'class_01/subject13_r0' to keep unique
        sample_dir = s['path'].replace('./Video_data/', '')  # 'class_01/subject13_r0'
        out_dir = os.path.join(OUT_ROOT, sample_dir)
        os.makedirs(out_dir, exist_ok=True)
        video = f'{VIDEO_ROOT}/{s["path"]}/sk_depth.avi'
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s['start'])
        n = s['end'] - s['start']
        written = 0
        for fi in range(n):
            ret, frame = cap.read()
            if not ret: break
            # depth is encoded as RGB (3-ch), convert to gray for depth modality
            # Actually MotionRGBD K-stream expects 3-ch depth — keep as-is
            h, w = frame.shape[:2]
            if (h, w) != (240, 320):
                frame = cv2.resize(frame, (320, 240))
            cv2.imwrite(os.path.join(out_dir, f'{fi:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        cap.release()
        split_lines.append(f'{sample_dir}/ {written} {s["label"]}\n')
        if i % 50 == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            eta = elapsed / max(1, i+1) * (len(samples) - i - 1)
            print(f'{split_name} {i+1}/{len(samples)} | {sample_dir} | {written} frames | elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)
    with open(f'{SPLITS_DIR}/{split_name}.txt', 'w') as f:
        f.writelines(split_lines)
    print(f'Wrote {SPLITS_DIR}/{split_name}.txt with {len(split_lines)} lines')

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    extract(test, 'valid')
elif len(sys.argv) > 1 and sys.argv[1] == 'train':
    extract(train, 'train')
else:
    extract(test, 'valid')
    extract(train, 'train')

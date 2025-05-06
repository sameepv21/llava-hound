import os
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Video Frame Sampler')
    parser.add_argument('--input_dir', type=str, help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, help='Output directory for sampled frames')
    return parser.parse_args()

import time

def sample_frames(video_path, output_dir):
    null_frame_count = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = total_frames // 10
    if sample_interval == 0:
        sample_interval = 1
    frame_count = 1
    for i in range(total_frames):
        ret, frame = cap.read()
        if sample_interval == 0:
            print(f'filename {video_path}')
        if i % sample_interval == 0:
            if frame is not None:
                cv2.imwrite(os.path.join(output_dir, f'c01_{frame_count:04d}.jpeg'), frame)
                frame_count += 1
            else:
                null_frame_count += 1
    cap.release()
    return null_frame_count

def main():
    args = parse_args()
    video_files = os.listdir(args.input_dir)
    null_frames = 0
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(args.input_dir, video_file)
        if video_file.endswith(".mp4"):
            video_id = os.path.splitext(video_file)[0]  # Extracting video_id from the file name
            output_dir = os.path.join(args.output_dir, video_id)
            os.makedirs(output_dir, exist_ok=True)
            null_frames_ind = sample_frames(video_path, output_dir)
            if null_frames_ind > 3:
                print(video_file)
                break
            null_frames += null_frames_ind

if __name__ == '__main__':
    main()
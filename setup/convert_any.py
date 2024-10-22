import os
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Video Frame Sampler')
    parser.add_argument('--input_dir', type=str, help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, help='Output directory for sampled frames')
    return parser.parse_args()

def sample_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = total_frames // 10
    frame_count = 1
    for i in range(total_frames):
        ret, frame = cap.read()
        if i % sample_interval == 0:
            cv2.imwrite(os.path.join(output_dir, f'c01_{frame_count:04d}.jpeg'), frame)
            frame_count += 1
    cap.release()

def main():
    args = parse_args()
    video_files = os.listdir(args.input_dir)
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(args.input_dir, video_file)
        video_id = os.path.splitext(video_file)[0]  # Extracting video_id from the file name
        output_dir = os.path.join(args.output_dir, video_id)
        os.makedirs(output_dir, exist_ok=True)
        sample_frames(video_path, output_dir)

if __name__ == '__main__':
    main()
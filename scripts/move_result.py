import os
import shutil
import argparse


parser = argparse.ArgumentParser('zpwang')
parser.add_argument('--cur_dir', type=str, default='/content/IDRR-base/output_space/')
parser.add_argument('--output_dir', type=str, default='/content/drive/MyDrive/IDRR/output_space')
args = parser.parse_args()

os.chdir(args.cur_dir)

for dirpath, dirnames, filenames in os.walk('.'):
    if 'checkpoint' in dirpath:
        continue
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        destination_path = os.path.join(args.output_dir, os.path.dirname(file_path))
        os.makedirs(destination_path, exist_ok=True)
        shutil.copy(file_path, destination_path)
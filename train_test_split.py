import os
import argparse
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Separates a CSV file into training and validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input_csv', metavar='input_csv',
        type=str,
        help='Path to the input CSV file')
    parser.add_argument('--image_path', help='images for train test split')
    parser.add_argument('--train_impath', help='images for train test split')
    parser.add_argument('--test_impath', help='images for train test split')
    parser.add_argument(
        '-f', metavar='train_frac',
        type=float,
        default=.75,
        help='fraction of the dataset that will be separated for training (default .75)')
    parser.add_argument(
        '-s', metavar='stratify',
        type=bool,
        default=True,
        help='Stratify by class instead of whole dataset (default True)')
    parser.add_argument(
        '-o', metavar='output_dir',
        type=str,
        default=None,
        help='Directory to output train and evaluation datasets (default input_csv directory)'
    )

    args = parser.parse_args()

    if args.f < 0 or args.f > 1:
        raise ValueError('train_frac must be between 0 and 1')

    # output_dir = input_csv directory is None
    if args.o is None:
        output_dir, _ = os.path.split(args.input_csv)
    else:
        output_dir = args.o

#     col_names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'height', 'width']  # use this header if you file has no header
#     df = pd.read_csv(args.input_csv, names=col_names)
    df = pd.read_csv(args.input_csv)
    
    strat = df['class'] if args.s else None
    train_df, validation_df = train_test_split(df, test_size=None, train_size=args.f, stratify=strat)
    
    if not os.path.exists(args.test_impath):
        os.makedirs(args.test_impath)
        
    if not os.path.exists(args.train_impath):
        os.makedirs(args.train_impath)
    
    for im_f in train_df['filename']:
        shutil.copyfile(os.path.join(args.image_path, im_f), os.path.join(args.train_impath, im_f))
    
    for im_f in validation_df['filename']:
        shutil.copyfile(os.path.join(args.image_path, im_f), os.path.join(args.test_impath, im_f))
    
    # output files have the same name of the input file, with some extra stuff appended
    new_csv_name = os.path.splitext(os.path.split(args.input_csv)[-1])[0]
    train_csv_path = os.path.join(output_dir, new_csv_name + '_train.csv')
    eval_csv_path = os.path.join(output_dir, new_csv_name + '_eval.csv')

    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(eval_csv_path, index=False)
    
    print(f'Train test split success, stratify={bool(strat)}, train_frac={args.f}')

    
    
    
#python train_test_split.py /home/ec2-user/obj-detection/annotations/coordinates.csv --image_path=/home/ec2-user/obj-detection/depot --train_impath=/home/ec2-user/obj-detection/train_images --test_impath=/home/ec2-user/obj-detection/eval_images -f 0.8 -o /home/ec2-user/obj-detection/annotations/
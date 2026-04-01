import os
import shutil
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for generating ImageNet-100')

    parser.add_argument('--source_folder', type=str,
     default='', help='folder of ImageNet-1K dataset')
    parser.add_argument('--target_folder', type=str,
     default='', help='folder of ImageNet-100 dataset')
    parser.add_argument('--target_class', type=str,
     default='IN100.txt', help='class file of ImageNet-100')

    opt = parser.parse_args()

    return opt

f = []
def generate_data(source_folder, target_folder, target_class):

    txt_data = open(target_class, "r") 
    for ids, txt in enumerate(txt_data):
        s = str(txt.split('\n')[0])
        f.append(s)

    # Loop over corruption type folders
    for corruption in os.listdir(source_folder):
        corruption_path = os.path.join(source_folder, corruption)
        if os.path.isdir(corruption_path):
            # Loop over severity levels 1-5
            for severity in os.listdir(corruption_path):
                severity_path = os.path.join(corruption_path, severity)
                if os.path.isdir(severity_path):
                    # Create corresponding folders in target directory
                    target_corruption = os.path.join(target_folder, corruption)
                    target_severity = os.path.join(target_corruption, severity)
                    os.makedirs(target_severity, exist_ok=True)
                    
                    # Loop over class folders
                    for dirs in os.listdir(severity_path):
                        for tg_class in f:
                            if dirs == tg_class:
                                print(f'{dirs} is transferred from {corruption}/{severity}')
                                src_path = os.path.join(severity_path, dirs)
                                dst_path = os.path.join(target_severity, dirs)
                                shutil.copytree(src_path, dst_path)


opt = parse_option()
generate_data(opt.source_folder, opt.target_folder, opt.target_class)


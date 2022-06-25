import os
import csv
import argparse
from pathlib import Path
import cv2
from PIL import Image
import os
import shutil
from cv2 import destroyAllWindows
import tifffile as tiff
from skimage import measure, transform
import numpy as np

def blurAlgo(source, target, copy, luv_flag=False, not_blur_plate=False, not_blur_face=False):
    copy_image = copy
    source_dir = source
    flag = luv_flag
    txts_dir = os.path.join(target,'labels')
    plate_list_path = os.path.join(target,'images_plate.txt')
    face_list_path = os.path.join(target,'images_face.txt')
    image_copy_dir = os.path.join(target,'blurred-images')

    if copy_image:
        Path(image_copy_dir).mkdir(parents = True, exist_ok=True)
    txts = os.listdir(txts_dir)
    img_names = []
    img_infos = []
    for txt in filter(lambda f: ".txt" in f, txts):
        name, ext = os.path.splitext(txt)
        img_names.append(name)

    #print(img_names)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            name, ext = os.path.splitext(file)
            if name in img_names:
                img_abs_path = os.path.join(root, file)
                img_infos.append((img_abs_path, name, ext))

    f_plate = open(plate_list_path, 'w', encoding='utf-8', newline='')
    f_face = open(face_list_path, 'w', encoding='utf-8', newline='')
    for img_abs_path, img_name, img_ext in img_infos:
        try:
                print('processing: ' + img_name + img_ext)
                face_written = False
                plate_written = False
                # output list
                txt_path = os.path.join(txts_dir, img_name + '.txt')
                for line in open(txt_path, "r").readlines():
                    label = int(line.split(' ')[0])             # 0:plate 1:face
                    if int(label) == 1 and not face_written:    
                        f_face.write(img_abs_path + '\n')
                        face_written = True
                    if int(label) == 0 and not plate_written:   
                        f_plate.write(img_abs_path + '\n')
                        plate_written = True
                           
                if not flag:              # blur RGB images
                    image = cv2.imread(img_abs_path)
                    image_blurred = image.copy()
                    width = image.shape[1] 
                    height = image.shape[0] 
                    labels = []
                    for line in open(txt_path, "r").readlines():
                        label = int(line.split(' ')[0])
                     # detect faces only
                        x = float(line.split(' ')[1])
                        y = float(line.split(' ')[2])
                        w = float(line.split(' ')[3]) * 0.85
                        h = float(line.split(' ')[4]) * 0.75

                        x_cv = int((x - w/2) * width) 
                        y_cv = int((y - h/2) * height)
                        h_cv = int(h * height) 
                        w_cv = int(w * width) 
                        k_cv = int(image.shape[0]/10)
                        
                        size = min(h_cv, w_cv)
                        block=(6, 6, 1)
                        if size < np.int8(16):
                            block = (4, 4, 1) 

                        downsample = measure.block_reduce(image[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv], block_size=block, func=np.max)
                        rows, columns = image[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv].shape[0], image[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv].shape[1]
                        upsample = cv2.resize(downsample, (columns, rows), interpolation = cv2.INTER_NEAREST)
                        image_blurred[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv] = upsample

                    write_file = os.path.join(image_copy_dir, img_name + img_ext)
                    print("saving to: " + write_file)
                    cv2.imwrite(write_file, image_blurred)

                elif flag:            # blur LUV images with tiff format         
                    luv_image = tiff.imread(img_abs_path)
                    image_blurred = luv_image.copy()
                    width = luv_image.shape[3] 
                    height = luv_image.shape[2]
                    for line in open(txt_path, "r").readlines():
                        label = int(line.split(' ')[0])
                      # detect faces only
                        x = float(line.split(' ')[1])
                        y = float(line.split(' ')[2])
                        w = float(line.split(' ')[3]) * 0.85
                        h = float(line.split(' ')[4]) * 0.75

                        x_cv = int((x - w/2) * width) 
                        y_cv = int((y - h/2) * height) 
                        h_cv = int(h * height) 
                        w_cv = int(w * width)

                        size = min(h_cv, w_cv)
                        # adjust the block size of downsampling (max-pooling)
                        block=(6, 6, 1)
                        if size < np.int8(16):
                            block = (4, 4, 1) 

                        if not_blur_plate and label == 0:
                            continue
                        elif not_blur_face and label == 1:
                            continue
                        else:
                            img0 = np.squeeze(luv_image)
                            img0 = np.rollaxis(img0, 0, 3)
                            downsample = measure.block_reduce(img0[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv], block_size=block, func=np.max)
                            rows, columns = img0[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv].shape[0], img0[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv].shape[1]
                            upsample = cv2.resize(downsample, (columns, rows), interpolation = cv2.INTER_NEAREST)
                            img0[y_cv:y_cv+h_cv, x_cv:x_cv+w_cv] = upsample
                            img1 = np.rollaxis(img0, 2, 0)
                            img1 = np.expand_dims(img1, axis=1)

                    write_file = os.path.join(image_copy_dir, img_name + img_ext)
                    tiff.imwrite(write_file, img1)
        except Exception as e:
            print(f"error by {img_name + img_ext}")
            print(e)
            continue

    f_plate.close()
    f_face.close()
    
    print('done')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= ".\\")
    parser.add_argument('--target', type=str, default='.\\target')
    parser.add_argument('--copy', type=bool, default=False)
    parser.add_argument('--luv-flag', action='store_true')
    opt = parser.parse_args()
    return opt

def main(opt):
    blurAlgo(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

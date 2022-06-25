# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:42:59 2021

@author: WEZ1CGD4
"""

from PIL import Image
import os 
 
def join(png1, png2,picture_name, flag='vertical'):
    """
    :param png1: path
    :param png2: path
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = Image.open(png1), Image.open(png2)
    # 统一图片尺寸，可以自定义设置（宽，高）
    img1 = img1.resize((1664, 512), Image.ANTIALIAS)
    img2 = img2.resize((1664, 512), Image.ANTIALIAS)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save('horizontal.png')
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1] + size2[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        
        file_path=os.path.join(os.path.abspath(os.curdir),"picture_joint")
        
        isExists=os.path.exists(file_path)
     
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(file_path)   
        save_path=os.path.join(file_path, picture_name)
        joint.save(save_path)
 
 
if __name__ == '__main__':
    clourpic_path=r"C:\Users\wez1cgd4\Desktop\face-plate-blurrer\Sample_Data"
    labelpic_path=r"C:\Users\wez1cgd4\Desktop\face-plate-blurrer\YOLO_output\output\blurred-images"
    filelist_clour=os.listdir(clourpic_path)
    filelist_label=os.listdir(labelpic_path)
     
    for file_label in filelist_label:
        filename=os.path.splitext(file_label)[0]
        file_temp=filename+'.jpeg'
        if file_label.endswith('.jpeg') and file_temp in filelist_clour:
            abspath_clour=os.path.join(clourpic_path, file_temp)
            abspath_label=os.path.join(labelpic_path, file_label)
    
            # 两张图片地址：
            png1 =abspath_clour
            png2 =abspath_label
            picture_name=file_label
            # 横向拼接
            # join(png1, png2, flag='vertical')
         
            # 纵向拼接
            join(png1, png2, picture_name,flag='vertical')
    print("completed,completed,completed!")
     
     
import os, sys
import argparse
import cv2, tifffile
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(ROOT)

if str(ROOT) + r'\yolov5_blur' not in sys.path:
    sys.path.append(str(ROOT) + r'\yolov5_blur')

from yolov5_blur import detect
from yolov5_blur.utils.general import print_args
import process

def parse_opt():
    parser = argparse.ArgumentParser()
    # arguments for detect.py
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_bosch.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256, 832], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # arguments for process.py
    parser.add_argument('--copy', action='store_true', help='blur and save images in the target directory')
    parser.add_argument('--luv-flag', action='store_true', help='the flag for blurring LUV or RGB images')
    parser.add_argument('--not-blur-plate', action='store_true', help='the flag for not blurring license plate')
    parser.add_argument('--not-blur-face', action='store_true', help='the flag for not blurring face')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    detect.run(weights=opt.weights, source=opt.source, save_txt=opt.save_txt, name=opt.name, nosave=opt.nosave, exist_ok=opt.exist_ok)
    process.blurAlgo(source=opt.source, target=opt.name, copy=opt.copy, luv_flag=opt.luv_flag)

    print("blur done")





from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
import glob
import shutil
import os
from tqdm import  tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    root = "/data-tmp/coco2017/val2017/*.jpg"

    if os.path.exists("./res"):
        shutil.rmtree("./res")
    os.makedirs("./res",exist_ok=True)

    imgs = glob.glob(root)
    for img_path in tqdm(imgs[::10]):
        result = inference_detector(model, img_path)
        # show the results
        img = show_result_pyplot(model, img_path, result, score_thr=args.score_thr)
        cv2.imwrite("./res/{}".format(os.path.basename(img_path)),img)


if __name__ == '__main__':
    main()

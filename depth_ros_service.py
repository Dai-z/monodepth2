from networks.depth_decoder import DepthDecoder
import rospy
import os
import sys
import argparse
import cv2
import numpy as np
from functools import partial

import networks

import torch
from torchvision import transforms

sys.path.append(
    os.path.join(os.environ["HOME"],
                 "Workspace/yn_ros/devel/lib/python3/dist-packages"))
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
from depth.srv import Depth, DepthRequest, DepthResponse
from cv_bridge import CvBridge


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_folder',
                        type=str,
                        help="name of model to load")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def handle_image(req: DepthRequest, encoder, decoder, feed_width, feed_height,
                 device):
    cv_bridge = CvBridge()

    image = cv_bridge.imgmsg_to_cv2(req.image)
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (feed_width, feed_height))
    # [H, W ,3] to [3, H, W]
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    features = encoder(image)
    outputs = decoder(features)
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width),
        mode="bilinear",
        align_corners=False)

    disp_resized = disp_resized.detach().cpu().squeeze().numpy()
    msg = cv_bridge.cv2_to_imgmsg(disp_resized.astype(np.float32))
    return DepthResponse(msg)


def main():
    rospy.init_node('depth_service')

    args = parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path = args.model_folder
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    # # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc,
                                          scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    handler = partial(handle_image,
                      encoder=encoder,
                      decoder=depth_decoder,
                      feed_height=feed_height,
                      feed_width=feed_width,
                      device=device)
    s = rospy.Service('depth_service', Depth, handler)
    rospy.spin()


if __name__ == "__main__":
    main()
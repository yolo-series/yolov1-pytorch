#!/usr/bin/env python3
# -----------------------------------------------------
# @Time : 2022/9/16
# @Author  : Firmin.Sun(fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import argparse
import torch
import onnx
import onnxsim

from backbone import resnet18

parser = argparse.ArgumentParser(description="Get config")
parser.add_argument('--output_path', help='output path', default='./output/resnet.onnx')
parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
args = parser.parse_args()

if __name__ == '__main__':
    print("found ", torch.cuda.device_count(), " GPU(s)")
    device = torch.device("cuda")
    model = resnet18(pretrained=True).to(device)
    model.eval()

    image = torch.randn(1, 3, 448, 448).to(device)

    try:
        torch.onnx.export(model, image, args.output_path, opset_version=10, verbose=False)

        # Checks
        onnx_model = onnx.load(args.output_path)  # load onnx net
        onnx.checker.check_model(onnx_model)  # check onnx net

        if args.simplify:
            try:
                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        onnx.save(onnx_model, args.output_path)

        print('ONNX export success, saved as %s' % args.output_path)
    except Exception as e:
        print('ONNX export failure: %s' % e)
"""
Copyright (c) <2019> <CEI Lab, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from api.proto.contrib import *

native_contrib_headers = \
    {"imagenet-proxy": lambda model_name: ImageNet_header(model_name, 32),
     "imagenet-96": lambda model_name: ImageNet_header(model_name, 96),
     "imagenet-128": lambda model_name: ImageNet_header(model_name, 128),
     "imagenet-160": lambda model_name: ImageNet_header(model_name, 160),
     "imagenet-192": lambda model_name: ImageNet_header(model_name, 192),
     "imagenet-224": lambda model_name: ImageNet_header(model_name, 224),
     "cifar10": lambda model_name: Cifar10_header(model_name)}

native_contrib_finals = \
    {"imagenet-proxy": lambda outstream_name : ImageNet_final(outstream_name, 32),
     "imagenet-96": lambda outstream_name : ImageNet_final(outstream_name, 96),
     "imagenet-128": lambda outstream_name: ImageNet_final(outstream_name, 128),
     "imagenet-160": lambda outstream_name: ImageNet_final(outstream_name, 160),
     "imagenet-192": lambda outstream_name: ImageNet_final(outstream_name, 192),
     "imagenet-224": lambda outstream_name: ImageNet_final(outstream_name, 224),
     "cifar10": lambda outstream_name: Cifar10_final(outstream_name)}


headers = {"native": native_contrib_headers}

finals = {"native": native_contrib_finals}

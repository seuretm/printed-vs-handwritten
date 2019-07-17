"""
OCR-D conformant command line interface
"""
import sys

from PIL import Image

from ..classifier import TypegroupsClassifier

def cli():
    """
    Run on sys.args
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Syntax: %s network-file image-file [stride]' % sys.argv[0])
        quit(1)
    network_file = sys.argv[1]
    image_file = Image.open(sys.argv[2])
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    classifier = TypegroupsClassifier(network_file, stride)
    result = classifier.run(image_file)
    print(result)

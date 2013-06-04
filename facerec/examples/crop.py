from facerec import ImageSet
import os
import sys

imgs = ImageSet(sys.argv[1])
crop = imgs.cropFaces()
crop.show()
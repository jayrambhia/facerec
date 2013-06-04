from facerec import ImageSet

imgs = ImageSet("/home/jay/FenchTose/Eigen/Brad")
imgs.show()
crop = imgs.cropFaces("/home/jay/SimpleCV/SimpleCV/Features/HaarCascades/face.xml")
crop.show()
print type(imgs), type(crop)
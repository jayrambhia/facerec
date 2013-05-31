import cv2
import warnings
import os
import collections
import numpy as np

class FaceRecognizer():

    def __init__(self):
        """
        Create a Face Recognizer Class using Fisher Face Recognizer. Uses
        OpenCV's FaceRecognizer class. Currently supports Fisher Faces.
        """
        self.supported = True
        self.model = None
        self.train_imgs = None
        self.train_labels = None
        self.csvfiles = []
        self.imageSize = None
        self.labels_dict = {}
        self.labels_set = []
        self.int_labels = []
        self.labels_dict_rev = {}
        if not hasattr(cv2, 'createFisherFaceRecognizer'):
            self.supported = False
            warnings.warn("Returning None. OpenCV >= 2.4.4 required.")
            return
        self.model = cv2.createFisherFaceRecognizer()

        # Not yet supported
        # self.eigenValues = None
        # self.eigenVectors = None
        # self.mean = None

    def train(self, images=None, labels=None, csvfile=None, delimiter=";"):
        """
        **SUMMARY**

        Train the face recognizer with images and labels.

        **PARAMETERS**

        * *images*    - A list of Images or ImageSet. All the images must be of
                        same size.
        * *labels*    - A list of labels(int) corresponding to the image in
                        images.
                        There must be at least two different labels.
        * *csvfile*   - You can also provide a csv file with image filenames
                        and labels instead of providing labels and images
                        separately.
        * *delimiter* - The delimiter used in csv files.

        **RETURNS**

        Nothing. None.

        **EXAMPLES**

        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)

        Save Fisher Training Data
        >>> f.save("trainingdata.xml")

        Load Fisher Training Data and directly use without trainging
        >>> f1 = FaceRecognizer()
        >>> f1.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f1.predict(imgs)

        Use CSV files for training
        >>> f = FaceRecognizer()
        >>> f.train(csvfile="CSV_file_name", delimiter=";")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        """

        if csvfile:
            images = []
            labels = []
            import csv
            try:
                f = open(csvfile, "rb")
            except IOError:
                warnings.warn("No such file found. Training not initiated")
                return None

            self.csvfiles.append(csvfile)
            filereader = csv.reader(f, delimiter=delimiter)
            for row in filereader:
                images.append(Image(row[0]))
                labels.append(row[1])

        if isinstance(labels, type(None)):
            warnings.warn("Labels not provided. Training not inititated.")
            return None

        self.labels_set = list(set(labels))
        i = 0
        for label in self.labels_set:
            self.labels_dict.update({label: i})
            self.labels_dict_rev.update({i: label})
            i += 1

        if len(self.labels_set) < 2:
            warnings.warn("At least two classes/labels are required"
                          "for training. Training not inititated.")
            return None

        if len(images) != len(labels):
            warnings.warn("Mismatch in number of labels and number of"
                          "training images. Training not initiated.")
            return None

        self.imageSize = images[0].shape[:2]
        h, w = self.imageSize
        images = [img if img.shape[:2] == self.imageSize 
                 else cv2.resize(img, (w, h)) for img in images]

        self.int_labels = [self.labels_dict[key] for key in labels]
        self.train_labels = labels
        labels = np.array(self.int_labels)
        self.train_imgs = images
        cv2imgs = [cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY) for img in images]

        self.model.train(cv2imgs, labels)
        # Not yet supported
        # self.eigenValues = self.model.getMat("eigenValues")
        # self.eigenVectors = self.model.getMat("eigenVectors")
        # self.mean = self.model.getMat("mean")

    def predict(self, imgs):
        """
        **SUMMARY**

        Predict the class of the image using trained face recognizer.

        **PARAMETERS**

        * *image*    -  Image.The images must be of the same size as provided
                        in training.

        **RETURNS**

        * *label* - Class of the image which it belongs to.

        **EXAMPLES**

        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)

        Save Fisher Training Data
        >>> f.save("trainingdata.xml")

        Load Fisher Training Data and directly use without trainging
        >>> f1 = FaceRecognizer()
        >>> f1.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f1.predict(imgs)

        Use CSV files for training
        >>> f = FaceRecognizer()
        >>> f.train(csvfile="CSV_file_name", delimiter=";")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None
        h, w = self.imageSize
        images = [img if img.shape[:2] == self.imageSize
                 else cv2.resize(img, (w, h)) for img in imgs]

        if isinstance(imgs, np.ndarray):
            if imgs.shape[:2] != self.imageSize:
                image = cv2.resize(imgs, (w, h))
            cv2img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            label, confidence = self.model.predict(cv2img)
            retLabel = self.labels_dict_rev.get(label)
            if not retLabel:
                retLabel = label
            return (retLabel, confidence)

        retVal = []
        for image in images:
            cv2img = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            label, confidence = self.model.predict(cv2img)
            retLabel = self.labels_dict_rev.get(label)
            if not retLabel:
                retLabel = label
            retVal.append((retLabel, confidence))
        return retVal

    # def update():
    #     OpenCV 2.4.4 doens't support update yet. It asks to train.
    #     But it's not updating it.
    #     Once OpenCV starts supporting update, this function should be added
    #     it can be found at https://gist.github.com/jayrambhia/5400347

    def save(self, filename):
        """
        **SUMMARY**

        Save the trainging data.

        **PARAMETERS**

        * *filename* - File where you want to save the data.

        **RETURNS**

        Nothing. None.

        **EXAMPLES**

        >>> f = FaceRecognizer()
        >>> imgs1 = ImageSet(path/to/images_of_type1)
        >>> labels1 = LabelSet("type1", imgs1)
        >>> imgs2 = ImageSet(path/to/images_of_type2)
        >>> labels2 = LabelSet("type2", imgs2)
        >>> imgs3 = ImageSet(path/to/images_of_type3)
        >>> labels3 = LabelSet("type3", imgs3)
        >>> imgs = concatenate(imgs1, imgs2, imgs3)
        >>> labels = concatenate(labels1, labels2, labels3)
        >>> f.train(imgs, labels)
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)

        #Save New Fisher Training Data
        >>> f.save("new_trainingdata.xml")
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None

        self.model.save(filename)

    def load(self, filename):
        """
        **SUMMARY**

        Load the trainging data.

        **PARAMETERS**

        * *filename* - File where you want to load the data from.

        **RETURNS**

        Nothing. None.

        **EXAMPLES**

        >>> f = FaceRecognizer()
        >>> f.load("trainingdata.xml")
        >>> imgs = ImageSet("path/to/testing_images")
        >>> print f.predict(imgs)        
        """
        if not self.supported:
            warnings.warn("Fisher Recognizer is supported by OpenCV >= 2.4.4")
            return None

        self.model.load(filename)
        loadfile = open(filename, "r")
        for line in loadfile.readlines():
            if "cols" in line:
                match = re.search("(?<=\>)\w+", line)
                tsize = int(match.group(0))
                break
        loadfile.close()
        w = int(tsize ** 0.5)
        h = tsize / w
        while(w * h != tsize):
            w += 1
            h = tsize / w
        self.imageSize = (w, h)

class ImageSet(list):

    def __init__(self, directory):
        try:
            imagefiles = os.listdir(directory)
            print imagefiles
        except OSError as error:
            warnings.warn("OS Error({0}): {1}".format(error.errno, error.strerror))
            return
        for imagefile in imagefiles:
            filename = os.path.join(directory, imagefile)
            img = cv2.imread(filename)
            #print img
            if isinstance(img, np.ndarray):
                #print self
                self.append(img)

class LabelSet(list):

    def __init__(self, label, imageset):
        if not isinstance(imageset, collections.Iterable):
            warnings.warn("The provided ImageSet is not a list")
            return None
        labels = [label]*len(imageset)
        self.extend(labels)

def concatenate(*args):
    retVal = []
    for arg in args:
        retVal.extend(arg)
    return retVal

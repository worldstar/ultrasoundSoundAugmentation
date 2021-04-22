import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import cv2,os
import pywt

class CustomDataGenerator(ImageDataGenerator):

    def __init__(self,
                fun="",
                clahenum=None,
                h=None,
                kernel=None,
                radius=None,
                n_points=None,
                templateWindowSize=None,
                searchWindowSize=None,
                num_clusters=None,

                **kwargs):
        '''
        Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        '''
        self.fun = fun #fun參數
        self.clahe_num = clahenum #clahe參數

        self.h = h #NLM參數
        self.templateWindowSize =templateWindowSize  # NLM參數
        self.searchWindowSize =searchWindowSize # NLM參數

        self.kernel = kernel #open參數
        self.radius =radius  # LBP算法中範圍半徑的取值
        self.n_points =n_points # 領域像素點數

        self.num_clusters = num_clusters #kmeans參數 獲得每個像素所屬的類別

        if self.fun == "NLM":
            function=self.NLM
        if self.fun == "CLAHE_Color":
            function=self.CLAHE_Color
        if self.fun == "Opening_operation":
            function=self.Opening_operation
        if self.fun == "OTSU":
            function=self.OTSU
        if self.fun == "normalize":
            function=self.normalize
        if self.fun == "HSVcolor":
            function=self.HSVcolor
        if self.fun == "LBP":
            function=self.LBP
        if self.fun == "CLAHENormalize":
            function=self.CLAHENormalize
        if self.fun == "CLAHEHSVcolor":
            function=self.CLAHEHSVcolor
        if self.fun == "NLMOpening":
            function=self.NLMOpening
        # if self.fun == "dwt":
        #     function=self.dwt
        if self.fun == "kmeans":
            function=self.kmeans

            
        
        super().__init__(
            preprocessing_function=function,
            **kwargs)
       

    def NLM(self, image):
        '''
        h:決定濾波器強度。較高的值可以更好的消除噪聲，但也會刪除圖像細節(10的效果比較好)
        hForColorComponents:與h相同，但只適用於彩色圖像(該值通常與h相同)
        templateWindowSize:奇數(推薦值為7)
        searchWindowSize:奇數(推薦值為21)
        '''
        temp=random.randint(0, 40)
        img = image.astype(np.uint8) # convert to int
        dst = cv2.fastNlMeansDenoisingColored(img,None,self.h,self.h,self.templateWindowSize,self.searchWindowSize)

        return dst

    def CLAHE_Color(self,image):

        img = image.astype(np.uint8) # convert to int
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return final

    def Opening_operation(self,image):

        kernel = np.ones((self.kernel,self.kernel),np.uint8) 
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        return opening
    
    def OTSU(self,image):

        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        return binary

    def normalize(self,image):
        img = image.astype(np.uint8)
        img_norm=cv2.normalize(img,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)


        return img_norm

    def HSVcolor(self,image):
        img = image.astype(np.uint8)
        HSVcolor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # cv2.imwrite('./test/normalize.png',HSVcolor)
        return HSVcolor

    def LBP(self,image):
        img = image.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp2 = local_binary_pattern(img, self.n_points, self.radius)
        # lbp2 =lbp2*255

        # cv2.imwrite('./test/normalize.png',img)
        return lbp2

    def CLAHENormalize(self,image):
        img = image.astype(np.uint8)
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        img_norm=cv2.normalize(final,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
        # cv2.imwrite('./test/clahenorm.png',img_norm)

        return img_norm

    def CLAHEHSVcolor(self,image):
        img = image.astype(np.uint8)
        HSVcolor = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab= cv2.cvtColor(HSVcolor, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(40,40))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # cv2.imwrite('./test/clahenorm.png',final)

        return final

    def NLMOpening(self,image):
        img = image.astype(np.uint8)
        dst = cv2.fastNlMeansDenoisingColored(img,None,self.h,self.h,self.templateWindowSize,self.searchWindowSize)
        kernel = np.ones((self.kernel,self.kernel),np.uint8) 
        opening = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite("./test/NLMOpening.png",opening)

        return opening

    def dwt(self,image):
        img = image.astype(np.uint8)
        img = cv2.resize(img, (256, 256))
        # 将多通道图像变为单通道图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # plt.figure('二维小波一级变换')
        coeffs = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs
        # # 将各个子图进行拼接，最后得到一张图
        AH = np.concatenate([cA, cH], axis=1)
        VD = np.concatenate([cV, cD], axis=1)
        img2 = np.concatenate([AH, VD], axis=0)

        # cv2.imwrite("./test/dwt.png",cH)

        return cH
                
    def kmeans(self,image):
        img = image.astype(np.uint8)
        h, w, ch = image.shape

        # 构建图像数据
        data = image.reshape((-1, 3))
        data = np.float32(data)

        # 图像分割
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # num_clusters = 4
        ret, label, center = cv2.kmeans(data, self.num_clusters, None, criteria, self.num_clusters, cv2.KMEANS_RANDOM_CENTERS)

        # 生成mask区域
        index = label[0][0]
        center = np.uint8(center)
        color = center[0]
        mask = np.zeros((h, w), dtype=np.uint8)
        label = np.reshape(label, (h, w))
        mask[label == index] = 255

        # 高斯模糊
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cv2.dilate(mask, se, mask)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # blurred = cv.GaussianBlur(mask, (5, 5), 0)
        # canny = cv.Canny(blurred, 30, 150)
        
        # 背景替换
        result = np.zeros((h, w, ch), dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                w1 = mask[row, col] / 255.0
                b, g, r = image[row, col]
                b = w1 * 0 + b * (1.0 - w1)
                g = w1 * 0 + g * (1.0 - w1)
                r = w1 * 0 + r * (1.0 - w1)
                result[row, col] = (b, g, r)

        # cv2.imwrite("./test/kmeans03.png",result)
        # cv.imshow("background-substitution", result)
        return result

                        
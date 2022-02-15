import cv2 as cv
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class Litofanias:
    # constructor
    def __init__(self, imagen):
        self.imagen_1 = imagen
        self.image_Corazon = cv.imread("Imagenes/Corazon.png")
        self.font_birthday = ImageFont.truetype("Fonts/BirthdayPU.ttf", 75)
    # Normal

    def WaterMark(self, image_Mark):
        img = cv.imread(self.imagen_1)
        img_Resize = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
        width_Original = img_Resize.shape[1]
        height_Original = img_Resize.shape[0]

        img_Marca = cv.imread(image_Mark)

        img_M_Resize = cv.resize(img_Marca, (100, 50),
                                 interpolation=cv.INTER_AREA)
        width_Marca = img_M_Resize.shape[1]
        height_Marca = img_M_Resize.shape[0]

        M = np.float32([[1, 0, 20], [0, 1, 20]])
        imageOut = cv.warpAffine(img_M_Resize, M, (500, 500))
        print(imageOut.shape[1])
        print(imageOut.shape[0])

        #imageOut = cv.bitwise_not(imageOut)
        imageOut = cv.bitwise_or(img_Resize, imageOut)
    # funcionales

    def Image_instantanea(self):
        org = (50, 470)
        fontScale = 1
        thickness = 1
        M = np.float32([[1, 0, 25], [0, 1, 25]])
        
        imagen_instantanea = cv.imread(self.imagen_1)
        print(imagen_instantanea.shape[0])
        print(imagen_instantanea.shape[1])
       # imagen_instantanea = cv.cvtColor(imagen_instantanea,cv.COLOR_BGR2GRAY)
        img_ins_res = cv.resize(imagen_instantanea, (375, 375), interpolation=cv.INTER_AREA)
        imageOut = cv.warpAffine(
            img_ins_res, M, (425, 525), borderValue=(35, 35, 35))
        return imageOut

    def Image_instantanea_txt(self, Texto, Pos):
        font = ImageFont.truetype("Fonts/BirthdayPU.ttf", 80)
        color = (255,255,255)
        fontScale = .5
        thickness = 1
        M = np.float32([[1, 0, 25], [0, 1, 25]])

        imagen_instantanea = cv.imread(self.imagen_1)
       # imagen_instantanea = cv.cvtColor(imagen_instantanea,cv.COLOR_BGR2GRAY)
        img_ins_res = cv.resize(
            imagen_instantanea, (375, 375), interpolation=cv.INTER_AREA)
        imageOut = cv.warpAffine(img_ins_res, M, (425, 525), borderValue=(35,35,35))
        cv2_im_rgb = cv.cvtColor(imageOut, cv.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        # use a truetype font
        font = ImageFont.truetype("Fonts/BirthdayPU.ttf", 80)
        # Draw the text
        draw.text(Pos, Texto, font=font, fill="#cbc5c3")
        # Get back the image to OpenCV
        cv2_im_processed = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
        return cv2_im_processed

    def Write_Scrable(self, Nombre, orientacion, position):

        imagenLit = cv.imread(self.imagen_1)
        if(orientacion == 'H'):
            for i in range(len(Nombre)):

                F_image = cv.imread('PiezasBorde/'+Nombre[i]+'.png')
                y_offset = position[0]
                x_offset = position[1] + (i*55)

                x_end = x_offset + F_image.shape[1]
                y_end = y_offset + F_image.shape[0]

                roi = imagenLit[y_offset:y_end, x_offset:x_end]

                b, _g, _r = cv.split(F_image)

                for i in range(F_image.shape[1]):
                    for j in range(F_image.shape[0]):
                        if b[j, i] >= 159:
                            F_image[j, i] = roi[j, i]
                        else:
                            F_image[j, i] = F_image[j, i]

                imagenLit[y_offset:y_end, x_offset:x_end] = F_image

        if(orientacion == 'V'):
            for i in range(len(Nombre)):
                F_image = cv.imread('PiezasBorde/'+Nombre[i]+'.png')

                x_offset = position[1]
                y_offset = position[0] + (i*55)

                x_end = x_offset + F_image.shape[1]
                y_end = y_offset + F_image.shape[0]

                roi = imagenLit[y_offset:y_end, x_offset:x_end]

                b, g, r = cv.split(F_image)

                for i in range(F_image.shape[1]):
                    for j in range(F_image.shape[0]):
                        if b[j, i] >= 159:
                            F_image[j, i] = roi[j, i]
                        else:
                            F_image[j, i] = F_image[j, i]

                imagenLit[y_offset:y_end, x_offset:x_end] = F_image

        return imagenLit

    def Corazon_Mensaje(self, Text):

        image_Corazon = cv.imread("Imagenes/Corazon.png")
        image = cv.resize(image_Corazon, (500, 500),
                          interpolation=cv.INTER_AREA)
        # Convert the image to RGB (OpenCV uss BGR)
        cv2_im_rgb = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        # use a truetype font
        font = ImageFont.truetype("Fonts/BirthdayPU.ttf", 80)
        # Draw the text
        draw.text((90, 180), Text, font=font, fill="#FFF")
        # Get back the image to OpenCV
        cv2_im_processed = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
        return cv2_im_processed

    def Imagen_txt(self, Texto, Position):
        image_Corazon = cv.imread(self.imagen_1)
        image = cv.resize(image_Corazon, (500, 500),
                          interpolation=cv.INTER_AREA)
        # Convert the image to RGB (OpenCV uss BGR)
        cv2_im_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        # use a truetype font
        font = ImageFont.truetype("Fonts/BirthdayPU.ttf", 35)
        # Draw the text
        draw.text(Position, Texto, font=font, fill="#FFF")
        # Get back the image to OpenCV
        cv2_im_processed = cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
        return cv2_im_processed

    def Spotify(self, Cancion, Autor):
        org = (50, 470)
        fontScale = 1
        thickness = 1
        M = np.float32([[1, 0, 25], [0, 1, 25]])
        font = ImageFont.truetype("Fonts/ProximaNovaAltBold.ttf", 80)
        imagen_instantanea = cv.imread(self.imagen_1)
        print(imagen_instantanea.shape)
       # imagen_instantanea = cv.cvtColor(imagen_instantanea,cv.COLOR_BGR2GRAY)
        img_ins_res = cv.resize(
            imagen_instantanea, (375, 375), interpolation=cv.INTER_AREA)
        imageOut = cv.warpAffine(
            img_ins_res, M, (425, 525), borderValue=(35, 35, 35))
        imageOut = cv.putText(img_ins_res, Cancion, org, font,
                              fontScale, (250, 250, 250), thickness, cv.LINE_AA)
        imageOut = cv.putText(imageOut, Autor, org, font,
                              fontScale, (250, 250, 250), thickness, cv.LINE_AA)
        return imageOut

    def Spotify(self, Code):
        self.Image_instantanea()
        spo = cv.imread(Code)
        spo = cv.bitwise_not(spo)

        y_offset = 0
        x_offset = 0
        x_end = x_offset + Code.shape[1]
        y_end = y_offset + Code.shape[0]

        self.imagen_1[y_offset:y_end, x_offset:x_end] = Code

        pass

    def Mosaico(self,im1,im2,im3):

        M = np.float32([[1, 0, 0], [0, 1, 0]])
        imagen_instantanea = cv.imread(self.imagen_1)
        img_ins_res = cv.resize(imagen_instantanea, (213, 213), interpolation=cv.INTER_AREA)
        
        img_1 = cv.imread(im1)
        img_1_r = cv.resize(img_1, (213, 213), interpolation=cv.INTER_AREA)
        img_2 = cv.imread(im2)
        img_2_r = cv.resize(img_2, (213, 213), interpolation=cv.INTER_AREA)
        img_3 = cv.imread(im3)
        img_3_r = cv.resize(img_3, (213, 213), interpolation=cv.INTER_AREA)
        

        imageOut = cv.warpAffine(img_ins_res, M, (426, 426))

        imageOut[213:img_1_r.shape[1]+213, 213:img_1_r.shape[0]+213] = img_1_r
        imageOut[0:img_1_r.shape[1], 213:img_1_r.shape[0]+213] = img_2_r
        imageOut[213:img_1_r.shape[1]+213, 0:img_1_r.shape[0]] = img_3_r

        return imageOut

    def Contorno(self):
        imagen = cv.imread(self.imagen_1)
        imagen = cv.resize(imagen, (500, 500), cv.INTER_AREA)
        im = cv.Canny(imagen, 150, 220)
        im = cv.bitwise_not(im)
        return im

    def CreateImage(self):
        cv.imwrite('Imagen_Lito.jpg',self.imagen_1)

    def Threshold(self):
        pass


class Functional_Images:
    def __init__(self, ima1):
        self.ima = ima1

    def Background_Delete(self):

        img = cv.imread(self.ima)
        blurred = cv.blur(img, (3, 3))
        canny = cv.Canny(blurred, 50, 200)

        # find the non-zero min-max coords of canny
        pts = np.argwhere(canny > 0)
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)

        # crop the region
        cropped = img[y1:y2, x1:x2]
        return cropped

    def rect_with_rounded_corners(image, r, t, c):
        """
        :param image: image as NumPy array
        :param r: radius of rounded corners
        :param t: thickness of border
        :param c: color of border
        :return: new image as NumPy array with rounded corners
        """

        c += (255, )

        h, w = image.shape[:2]

        # Create new image (three-channel hardcoded here...)
        new_image = np.ones((h+2*t, w+2*t, 4), np.uint8) * 255
        new_image[:, :, 3] = 0

        # Draw four rounded corners
        new_image = cv.ellipse(
            new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
        new_image = cv.ellipse(
            new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
        new_image = cv.ellipse(
            new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
        new_image = cv.ellipse(
            new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)

        # Draw four edges
        new_image = cv.line(new_image, (int(r+t/2), int(t/2)),
                            (int(w-r+3*t/2-1), int(t/2)), c, t)
        new_image = cv.line(new_image, (int(t/2), int(r+t/2)),
                            (int(t/2), int(h-r+3*t/2)), c, t)
        new_image = cv.line(new_image, (int(r+t/2), int(h+3*t/2)),
                            (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
        new_image = cv.line(new_image, (int(w+3*t/2), int(r+t/2)),
                            (int(w+3*t/2), int(h-r+3*t/2)), c, t)

        # Generate masks for proper blending
        mask = new_image[:, :, 3].copy()
        mask = cv.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
        mask[mask != 128] = 0
        mask[mask == 128] = 1
        mask = np.stack((mask, mask, mask), axis=2)

        # Blend images
        temp = np.zeros_like(new_image[:, :, :3])
        temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
        new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask

        # Set proper alpha channel in new image
        temp = new_image[:, :, 3].copy()
        new_image[:, :, 3] = cv.floodFill(
            temp, None, (int(w/2+t), int(h/2+t)), 255)[1]

        return new_image

class Cartoonizer:
    """Cartoonizer effect  
        A class that applies a cartoon effect to an image.  
        The class uses a bilateral filter and adaptive thresholding to create  
        a cartoon effect.  
    """

    def __init__(self):
        pass

    def render(self, img_rgb):
        img_rgb = cv.imread(img_rgb)
        img_rgb = cv.resize(img_rgb, (600, 600))
        numDownSamples = 2   # number of downscaling steps
        numBilateralFilters = 25  # number of bilateral filtering steps
        # -- STEP 1 --
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv.pyrDown(img_color)
        cv.imshow("downcolor", img_color)
        # cv2.waitKey(0)
        # repeatedly apply small bilateral filter instead of applying
        # one large filter
        for _ in range(numBilateralFilters):
            img_color = cv.bilateralFilter(img_color, 9, 9, 7)
        cv.imshow("bilateral filter", img_color)
        # cv2.waitKey(0)
        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv.pyrUp(img_color)
        cv.imshow("upscaling", img_color)
        # cv2.waitKey(0)
        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_blur = cv.medianBlur(img_gray, 3)
        cv.imshow("grayscale+median blur", img_color)

        # cv2.waitKey(0)
        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv.adaptiveThreshold(img_blur, 255,
                                        cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY, 9, 2)
        cv.imshow("edge", img_edge)
        # cv2.waitKey(0)
        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        (x, y, z) = img_color.shape
        img_edge = cv.resize(img_edge, (y, x))
        img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2RGB)
        cv.imwrite("edge.png", img_edge)
        cv.imshow("step 5", img_edge)
        cv.waitKey(0)
        (x, y, z) = img_color.shape
        img_edge = cv.resize(img_edge, (y, x))
        return cv.bitwise_and(img_color, img_edge)

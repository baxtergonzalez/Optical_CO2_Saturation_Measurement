import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, name, blur = 5, method = 'Gaussian', lower_bound = 120, upper_bound = 250, central_percentage = .2):
        path = os.path.join('Sample_Images/Original_Images', name)
        self.original_image = cv2.imread(path)

        self.name = name

        self.image = np.copy(self.original_image)
        self.crop(central_percentage)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.output_images = []
        # self.output_images.append(['original', self.original_image])
        self.output_images.append(['resized', self.image])

        self.preprocessed_img = self.preprocess(blur = blur, method = method, lower_bound = lower_bound, upper_bound = upper_bound)
        self.edge_detect(canny1 = 100, canny2 = 200)
        # self.find_contours(max_area = 500)
        self.find_contours_alt()
        # self.color_mask(self.image)
        self.color_mask(self.image, lower_bound = (158, 84, 25), upper_bound = (179, 255, 255)) #Color mask for pink example

        #display outer contour and color masked contour
        combo_image = np.zeros_like(self.gray)
        cv2.drawContours(combo_image, self.outer_contour, -1, (255, 255, 255), 3)
        cv2.drawContours(combo_image, self.color_contour, -1, (255, 255, 255), 3)

        outer_area = cv2.contourArea(self.outer_contour)
        color_area = cv2.contourArea(self.color_contour)


        print(f'Area of outer contour: {outer_area}px^2\nArea of color contour: {color_area}px^2\nArea not covered by color: {outer_area - color_area}px^2\nPercentage of color: {color_area / outer_area * 100}%')

        self.output_images.append(['combo', combo_image])

        self.find_penetration_depth()

    def crop(self, percentage):
        x_min = int(self.image.shape[1] * percentage)
        x_max = int(self.image.shape[1] * (1 - percentage))
        y_min = int(self.image.shape[0] * percentage)
        y_max = int(self.image.shape[0] * (1 - percentage))

        self.image = self.image[y_min:y_max, x_min:x_max]

    def find_lines(self):
        self.lines_image = np.zeros_like(self.image)
        self.output_images.append(['lines', self.lines_image])
        self.lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, 100, minLineLength = 50, maxLineGap = 10)
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.output_images.append(['lines', self.lines_image])
        
    def sharpen(self, image, severity = 2):
        if severity == 1:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        elif severity == 2:
            kernel = (1/9)*np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]])
        elif severity == 3:
            kernel = np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]])
        else:
            raise ValueError('Invalid severity')
        
        return cv2.filter2D(image, -1, kernel)
    
    def blur(self, image, blur = 5, method = 'Gaussian'):
        if method == 'Gaussian':
            return cv2.GaussianBlur(image, (blur, blur), 0)
        elif method == 'Median':
            return cv2.medianBlur(image, blur)
        elif method == 'Bilateral':
            return cv2.bilateralFilter(image, blur, 75, 75)
        else:
            raise ValueError('Invalid blur method')

    def threshold(self, image, lower_bound = 120, upper_bound = 250):
        ret, thresh = cv2.threshold(image, lower_bound, upper_bound, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #Invert the image
        thresh = cv2.bitwise_not(thresh)

        return thresh

    def morphological_close(self, image, kernel_size = 3, num_iterations = 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        eroded = cv2.morphologyEx(image, cv2.MORPH_DILATE,kernel, iterations = num_iterations)
        blurred = self.blur(eroded, blur = 15, method = 'Gaussian')
        thresholded = self.threshold(blurred, lower_bound = 145, upper_bound = 255)
        thinned = cv2.morphologyEx(thresholded, cv2.MORPH_ERODE, kernel*3, iterations = num_iterations)

        return thinned

    def preprocess(self, blur = 5, method = 'Gaussian', lower_bound = 200, upper_bound = 250):
        gray = self.blur(self.gray, blur, method)
        gray = self.sharpen(gray, severity = 2)
        gray = self.threshold(gray, lower_bound, upper_bound)
        gray = self.morphological_close(gray, kernel_size = 3, num_iterations = 3)

        self.output_images.append(['preprocessed', gray])
        return gray

    def find_corners(self, blur = 5, method = 'Gaussian', lower_bound = 120, upper_bound = 250):
        corner_image = np.copy(self.image)
        corners = cv2.goodFeaturesToTrack(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2GRAY), 4, 0.5, 100)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corner_image, (x, y), 20, 255, -1)
        self.output_images.append(['corners', corner_image])

    def edge_detect(self, canny1 = 100, canny2 = 200):
        gray = self.preprocessed_img
        self.edges = cv2.Canny(gray, canny1, canny2)
        # combined = self.image + cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
        self.closed_edges = self.morphological_close(image = self.edges, kernel_size = 3, num_iterations = 1)
        self.closed_edges = cv2.bitwise_not(self.closed_edges)

        self.output_images.append(['edges',self.edges])
        self.output_images.append(['closed_edges', self.closed_edges])
        # self.output_images.append(['combined',combined])

    def color_mask(self, image, lower_bound = (136, 0, 136), upper_bound = (168, 255, 255)):
        img = np.copy(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        target = cv2.bitwise_and(hsv, hsv, mask = mask)

        #find contours in the mask and save
        color_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.color_contour = max(color_contours, key = cv2.contourArea)

        # target = cv2.cvtColor(target, cv2.COLOR_HSV2BGR)
        # cv2.drawContours(target, [self.color_contour], -1, (255, 255, 0), 3)
        # # self.output_images.append(['color_mask', target])
        # combined = np.copy(cv2.cvtColor(self.contour_image, cv2.COLOR_GRAY2BGR)) + target
        # self.output_images.append(['combined', combined])

    def find_contours(self, max_area = 10000, central_percentage = 0.2):
        # self.edge_detect()
        # contours, hierarchy = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(self.closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(self.gray)

        usable_contours = []


        for contour in contours:
            cnt_len = cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, 0.02 * cnt_len, True)
            if cv2.contourArea(approx_contour) >= max_area:
                usable_contours.append(approx_contour)

        max_contour = 0
        for contour in contours:
            if cv2.contourArea(contour) > max_contour:
                max_contour = cv2.contourArea(contour)
                final_contour = contour
        # cv2.drawContours(contour_image, [final_contour], -1, (0, 255, 255), 2)       

        # cv2.drawContours(contour_image, hull_list, -1, (255, 255, 255), 2)
        cv2.drawContours(contour_image, usable_contours, -1, (255, 0, 255), 1)
        # cv2.drawContours(contour_image, contours, -1,(255, 255, 0), 1)
        # cv2.drawContours(contour_image, central_hulls, -1, (0, 255, 0), 3)

        self.contour_image = contour_image

        self.output_images.append(['contours', contour_image])

    def find_contours_alt(self):
        #find contours on closed edges
        contours, hierarchy = cv2.findContours(self.closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.outer_contour = max(contours, key = cv2.contourArea)
        
        
        self.contour_image = np.zeros_like(self.gray)
        cv2.drawContours(self.contour_image, contours, -1, (255, 255, 255), 1)
        cv2.fillPoly(self.contour_image, contours, (255, 255, 0))

        print(f'{len(contours)} contours found')

        self.output_images.append(['closed contours', self.contour_image])

    def find_penetration_depth(self):
        #Find the centroid of the outer contour
        M = cv2.moments(self.outer_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        #Translate both contours so that centroid is at origin
        translated_outer_cont = self.outer_contour - [cx, cy]
        translated_color_cont = self.color_contour - [cx, cy]

        polar_outer_cont = []
        polar_color_cont = []
 
        #Convert contours to polar coordinates
        for outer_pt in translated_outer_cont:
            pt = np.array([outer_pt[0][0], outer_pt[0][1]])
            r = np.linalg.norm(pt)
            theta = np.arctan2(pt[1], pt[0])
            polar_outer_cont.append([r, theta])

        for color_pt in translated_color_cont:
            pt = np.array([color_pt[0][0], color_pt[0][1]])
            r = np.linalg.norm(pt)
            theta = np.arctan2(pt[1], pt[0])
            polar_color_cont.append([r, theta])

        #Sort contours by angle
        polar_outer_cont = sorted(polar_outer_cont, key = lambda x: x[1])
        polar_color_cont = sorted(polar_color_cont, key = lambda x: x[1])

        self.graph_depth(polar_outer_cont, polar_color_cont)

    def graph_depth(self, contour_1, contour_2):
        c1_theta = [x[1] for x in contour_1]
        c1_r = [x[0] for x in contour_1]

        c2_theta = [x[1] for x in contour_2]
        c2_r = [x[0] for x in contour_2]

        # difference_r = np.array(c2_r) - np.array(c1_r)
        
        plt.plot(c1_theta, c1_r, 'r')
        plt.plot(c2_theta, c2_r, 'b')
        # plt.plot(c2_theta, difference_r, 'g')

    def show(self):
        scale_factor = 0.25
        new_size = (int(self.image.shape[1] * scale_factor), int(self.image.shape[0] * scale_factor))
        
        for image in self.output_images:
            resize_image = cv2.resize(image[1], new_size)
            cv2.imshow(image[0], resize_image)

        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
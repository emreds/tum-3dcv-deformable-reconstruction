import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('../dataset/coke/image.png')
img2 = img.copy()
template = cv.imread('../dataset/coke/template_3.png')
#print(template.shape)
w, h = template.shape[:2][::-1]


templateR, templateG, templateB = cv.split(template)
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    imageMainR, imageMainG, imageMainB = cv.split(img)
    method = eval(meth)
    # Apply template Matching
    #res = cv.matchTemplate(img,template,method)
    resultB = cv.matchTemplate(imageMainR, templateR, method)
    resultG = cv.matchTemplate(imageMainG, templateG, method)
    resultR = cv.matchTemplate(imageMainB, templateB, method)
    res = resultR + resultG + resultB
    min_valr, max_valr, min_locr, max_locr = cv.minMaxLoc(resultR)
    min_valg, max_valg, min_locg, max_locg = cv.minMaxLoc(resultG)
    min_valb, max_valb, min_locb, max_locb = cv.minMaxLoc(resultB)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # print(min_locr + min_locg)
    top_left = [0,0]
    print(f'min_locr: {min_locr}')
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        
        #top_left = min_locr + min_locg + min_locb
        top_left[0] = min_locr[0] + min_locg[0] + min_locb[0]
        top_left[1] = min_locr[1] + min_locg[1] + min_locb[1]
    else:
        #top_left = max_locr +  max_locg +  max_locb
        top_left[0] = max_locr[0] + max_locg[0] + max_locb[0]
        top_left[0] = max_locr[1] + max_locg[1] + max_locb[1]
        
        
    print(f'top left {top_left}')
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    #print(top_left)
    #print(bottom_right)
    cv.rectangle(img, top_left, bottom_right, (255, 255, 255), )
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
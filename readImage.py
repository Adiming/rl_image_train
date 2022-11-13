import os
import cv2
import numpy as np

image_path = os.path.join(os.getcwd(),"images")

x = 0
y = 12

img_name = str(x) + "_" + str(y) + '.png'
img_path = os.path.join(image_path, img_name)
img = cv2.imread(img_path)
h,w,_ = img.shape

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#转换为灰色gray_img
cv2.imshow('gray_img',gray_img)


#对图像二值化处理 输入图像必须为单通道8位或32位浮点型
ret,thresh = cv2.threshold(gray_img,127,255,0)
cv2.imshow('thresh',thresh)



#寻找图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
# image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
image,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('image',image)
# print('contours[0]:',contours[0])
# print('len(contours)',len(contours))
# print('hierarchy,shape',hierarchy.shape)
# print('hierarchy[0]:',hierarchy[0])


#在原图img上绘制轮廓contours
# img = cv2.drawContours(img,contours,-1,(0,255,0),2)
# cv2.imshow('contours',img)

# img = cv2.resize(img, (84,84), interpolation= cv2.INTER_LINEAR) # resize the image

# edges = cv2.Canny(img,100,200)
# print(img)

#150 代表应该检测到的行的最小长度
# lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

# print(lines)
# for i in range(len(lines)):
#     for rho,theta in lines[i]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + w*(-b))
#         y1 = int(y0 + w*(a))
#         x2 = int(x0 - w*(-b))
#         y2 = int(y0 - w*(a))
     
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

#最低线段的长度，小于这个值的线段被抛弃
# minLineLength = 50
# minLineLength = 30

#线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
# maxLineGap =50
# maxLineGap =25

#100阈值，累加平面的阈值参数，即：识别某部分为图中的一条直线时它在累加平面必须达到的值，低于此值的直线将被忽略。
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 3, minLineLength, maxLineGap)

# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    
# cv2.imshow("lines", img)
# cv2.imshow("edges", edges)

cv2.waitKey()
cv2.destroyAllWindows()

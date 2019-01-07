import matplotlib.pyplot as plt
import numpy as np
import cv2

'''Erosion Method'''
def erosion(image, kernel):
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    h = kernel_height//2
    w = kernel_width//2
    
    res = [[0 for x in range(img_width)] for y in range(img_height)] 
    res = np.array(res)
    for i in range(h, img_height-h):
        for j in range(w, img_width-w):
            a = np.array(image[(i-h):(i-h)+kernel_height, (j-w):(j-w)+kernel_width])
            if(np.array_equal(a, kernel)):
                res[i][j] = 1
            else:
                res[i][j] = 0
    return res

'''Point Detection Method'''
def point_detection(image, kernel):
    img_height = image.shape[0]
    img_width = image.shape[1]
    image = cv2.Laplacian(image, cv2.CV_32F)
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    h = kernel_height//2
    w = kernel_width//2
    '''Threshold chosen to be a value which is 90% of maximum sum value'''
    T = 8382 
    sum_arr = []
    res = [[0 for x in range(img_width)] for y in range(img_height)] 
    res = np.array(res)
    for i in range(h, img_height-h):
        for j in range(w, img_width-w):
            a = np.array(image[(i-h):(i-h)+kernel_height, (j-w):(j-w)+kernel_width])
            out = ((np.multiply(kernel, a)))
            sum = np.abs(np.sum(out))
            sum_arr.append(sum)
            if(sum > T):
                co_ord = (i, j) 
                res[i][j] = 1
    print("Maximum sum: ",np.max(np.array(sum_arr)))
    return res, co_ord

def check_segment(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    '''Threshold chosen by observing the plotted histogram'''
    T = 204 
    res = [[0 for x in range(img_width)] for y in range(img_height)]
    res = np.array(res)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j] > T):
                res[i][j] = 255
            else:
                res[i][j] = 0
    return res
    
img = cv2.imread("point.jpg",0)
sample = img

kernel = np.array([[-1,-1,-1],
                    [-1,8,-1],
                    [-1,-1,-1]])

output, co_ord = point_detection(img, kernel)
output = output*255
output = np.asarray(output, np.uint8)
cv2.rectangle(output,(424,230),(464,272),(255,255,255),2)
cv2.imwrite("res_point.jpg",output)

'''Code for segmenting the object from the background'''
img2 = cv2.imread("segment.jpg", 0)
seg = check_segment(img2)
seg = np.asarray(seg, np.uint8)
cv2.rectangle(seg,(155,115),(208,172),(255,255,255),2)
cv2.rectangle(seg,(245,68),(300,223),(255,255,255),2)
cv2.rectangle(seg,(322,13),(370,291),(255,255,255),2)
cv2.rectangle(seg,(382,33),(430,264),(255,255,255),2)

'''Observed co-ordinates of bounding boxes, in col, row format'''
print("1st box: ")
print("Upper left: (155,115)")
print("Upper right: (208,115)")
print("Bottom left: (155,172)")
print("Bottom right: (208,172)\n")

print("2nd box: ")
print("Upper left: (245,68)")
print("Upper right: (300,68)")
print("Bottom left: (245,223)")
print("Bottom right: (300,223)\n")

print("3rd box: ")
print("Upper left: (322,13)")
print("Upper right: (370,13)")
print("Bottom left: (322,291)")
print("Bottom right: (370,291)\n")

print("4th box: ")
print("Upper left: (382,33)")
print("Upper right: (430,33)")
print("Bottom left: (382,264)")
print("Bottom right: (430,264)")

cv2.imwrite("res_segment.jpg",seg)

'''Plotting Histogram'''
my_dict = {}
for i in range(np.unique(img2).shape[0]):
    a = np.unique(img2)[i]
    count = np.sum(img2 == a)
    my_dict[a] = count
    
sorted_by_value = sorted(my_dict.items(), key=lambda kv: kv[1])
uniq = list(np.unique(img2))
val = list(my_dict.values())
plt.plot(uniq[1:],val[1:])
plt.show()

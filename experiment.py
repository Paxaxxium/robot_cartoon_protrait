import time, traceback, sys, cv2
import onnx
from portrait import AnimeGANv3, FaceSegmentation
import torchvision.transforms as transforms
import torch
import facer
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import root
from concave_hull import concave_hull, concave_hull_indexes

def ver_comparator(input):
    return input[1]
def hor_comparator(input):
    return input[0]

def angle_with_x_axis(point, reference_point):
    x_diff = point[0] - reference_point[0]
    y_diff = point[1] - reference_point[1]
    rad =  math.atan2(y_diff, x_diff)
    angle = math.degrees(rad) % 360
    # print(angle)
    if 180 <= angle < 270:
        return -1*rad + (3*np.pi/2)
    elif 90 <= angle < 180:
        return rad
    elif 0 <= angle < 90:
        return -1*rad  # First quadrant
    else:
        return rad - (3*np.pi/2)  # Fourth quadrant

def find_centroid(points):
    x_coords = points[:,0]; y_coords = points[:,1]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return (centroid_x, centroid_y)
# x = np.array([280.0692, 228.6441])  # X coordinates of the two points
# y = np.array([368.0476, 211.8257])  # Y coordinates of the two points

# coefficients = np.polyfit(x, y, 2)  # Fit a polynomial of degree 2 (parabola) to the points
# x_values = np.linspace(np.min(x), np.max(x), 100)
# y_values = np.polyval(coefficients, x_values)
# plt.scatter(x, y, color='red', label='Points')
# plt.plot(x_values, y_values, label='Fitted Parabola')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

# image_path = "C:\\Users\\Peijing Xu\\projects\\wen_research\\robotic_portrait\Screenshot 2024-11-24 103226.png"
# image = Image.open(image_path)

# Preprocess the image

def eye_drawing(eye,brow,img,position):
    # x_brow = brow[:,0]
    # y_brow = brow[:,1]
    traj1,traj3 = [],[]
    if(len(brow) != 0):
        v_indices = np.argsort(np.apply_along_axis(ver_comparator, 1, brow))
        h_indices = np.argsort(np.apply_along_axis(hor_comparator, 1, brow))
        vertical_brow = brow[v_indices]
        horizontal_brow = brow[h_indices]
        left,right = horizontal_brow[0],horizontal_brow[-1]
        bottom,top = vertical_brow[0], vertical_brow[-1]
        diff = np.sqrt(abs(top[1] - bottom[1]))
        coeff1 = np.polyfit([left[0],bottom[0]],[left[1],bottom[1]],1)
        coeff2 = np.polyfit([bottom[0],right[0]],[bottom[1],right[1]],1)
        coeff3 = np.polyfit([left[0],bottom[0],right[0]],[left[1],bottom[1]+diff,right[1]],3)
        left_func = np.poly1d(coeff1)
        right_func = np.poly1d(coeff2)
        poly_func = np.poly1d(coeff3)
        x1 = np.arange(left[0],bottom[0],0.1)
        x2 = np.arange(bottom[0],right[0]+1,0.1)
        x = np.concatenate((x1[:-1],x2))
        y1 = left_func(x1)
        y2 = right_func(x2)
        y3 = poly_func(x)
        plt.plot(x1,y1,color='black',linewidth=2)
        plt.plot(x2,y2,color='black',linewidth=2)
        plt.plot(x,y3,color='black',linewidth=1)
        traj1 = np.concatenate((np.column_stack((x1,y1)),np.column_stack((x2,y2))))
        traj3 = np.column_stack((x,y3))
                
    # j,k = 0,0
    # while(j < len(x)):
    #     img.putpixel((x[j],round(y3[j])),(0,0,0))
    #     if(j < len(x1)):
    #         img.putpixel((x1[j],round(y1[j])),(0,0,0))
    #     else:
    #         img.putpixel((x2[k],round(y2[k])),(0,0,0))
    #         k += 1
    #     j += 1
    # x_eye = eye[:,0]
    # y_eye = eye[:,1]
    top_traj,bottom_traj = [],[]
    hull = ConvexHull(eye)
    vertices = eye[hull.vertices]
    sorted_indices = np.argsort(np.apply_along_axis(ver_comparator, 1, vertices))
    sorted_v = np.array(vertices[sorted_indices])
    center = sorted_v[0][1] +  ((sorted_v[-1][1] - sorted_v[0][1])/2)
    lower,upper = [],[]
    i = 0
    while(sorted_v[i][1] <= center):
        lower.append(sorted_v[i])
        i+=1
    upper = np.array((sorted_v[i:]))
    lower = np.array((lower))
    lower_indices = np.argsort(np.apply_along_axis(hor_comparator, 1, lower))
    upper_indices = np.argsort(np.apply_along_axis(hor_comparator, 1, upper))
    lower = lower[lower_indices]
    upper = upper[upper_indices]

    upp_func = interp1d(upper[:,0],upper[:,1],fill_value="extrapolate")
    low_func = interp1d(lower[:,0],lower[:,1],fill_value="extrapolate")

    mi,mx = min(eye[:,0]), max(eye[:,0])
    x = np.arange(mi,mx,0.1)
    y1,y2 = upp_func(x),low_func(x)
    non_nan_indices1 = np.argwhere(~np.isnan(y1)).flatten()
    non_nan_indices2 = np.argwhere(~np.isnan(y2)).flatten()

    plt.plot(x[non_nan_indices1],y1[non_nan_indices1],color='black')
    plt.plot(x[non_nan_indices2],y2[non_nan_indices2],color='black',linewidth=2)

    top_traj = np.column_stack((x[non_nan_indices1],y1[non_nan_indices1]))
    bottom_traj = np.column_stack((x[non_nan_indices2],y2[non_nan_indices2]))

    midd_x = (mi + (abs(mx - mi) // 2))
    # y1 = np.array(y1)
    # y2 = np.array(y2)
    r =  (abs(y2[len(x)//2] - y1[len(x)//2]) / 2)
    midd_y = y2[len(x)//2] + r
    # img.putpixel((midd_x,midd_y),(0,0,0))
    x_circle = np.linspace(midd_x - r, midd_x + r, 50)
    y_circle_top = upp_func(x_circle)
    y_circle_bottom = low_func(x_circle)
    plt.fill_between(x_circle,y_circle_top,y_circle_bottom,color='black')
    # y_top = np.sqrt(r - (x_circle - midd_x)**2) + midd_y
    # y_bottom = -np.sqrt(r - (x_circle - midd_x)**2) + midd_y
    # plt.plot(x_circle,y_top,color='black')
    # plt.plot(x_circle,y_bottom,color='black')
    # if(position == 'left'):
    #     X = np.arange(mi+5,mx-5,1)
    #     for i in range(len(5)):
            
    plt.imshow(img)
    return traj1,traj3,top_traj,bottom_traj
    # for i in range(len(x)):
    #     img.putpixel((x[i],round(y1[i])),(0,0,0))
    #     img.putpixel((x[i],round(y2[i])),(0,0,0))

def nose_drawing(nose,pt,img):
    v_indices = np.argsort(np.apply_along_axis(ver_comparator, 1, nose))
    ver_nose = nose[v_indices]
    hull = ConvexHull(nose)
    vertices = nose[hull.vertices]
    m,b = np.polyfit([pt[0],ver_nose[0][0]],[pt[1],ver_nose[0][1]],1)
    deg = np.degrees(np.tan(m))
    right = False #right for the perspective of the face in the drawing
    if((-90 > deg >= -180) or (90 > deg >= 0)):
        right = True
    # v = []
    h = []
    for i in range(len(vertices)):
        if(vertices[i][1] > pt[1]):
            h.append(vertices[i])
    h = np.array(h)
    # top,next = [0,0],[0,0]
    # if(right):
    #     for i in range(len(vertices)):
            # if(m*vertices[i][0] + b >= vertices[i][1] or vertices[i][1] > pt[1]):
            #     if(top[1] > vertices[i][1] or top[1] == 0):
            #         top = vertices[i]
            #     if(next[0] > vertices[i][0] or next[0] == 0):
            #         next = vertices[i]
            #         v.append(vertices[i])
    #         if(vertices[i][1] > pt[1]):
    #             h.append(vertices[i])
    #     v = np.array(v)
    #     h = np.array(h)
    # else:
    #     for i in range(len(vertices)):
            # if(m*vertices[i][0] + b < vertices[i][1] or vertices[i][1] > pt[1]):
            #     if(top[1] > vertices[i][1] or top[1] == 0):
            #         top = vertices[i]
            #     if(next[0] < vertices[i][0] or next[0] == 0):
            #         next = vertices[i]
            #         v.append(vertices[i])
        #     if(vertices[i][1] > pt[1]):
        #         h.append(vertices[i])
        # v = np.array(v)
        # h = np.array(h)
    #diff = next[0] - (next[1] - b)/m
    # tmp = (top[1] - b)/m #+ diff
    # x = np.arange(min(next[0],tmp),max(next[0],tmp),0.01)
    # # plt.plot(x, m*(x - diff) + b,color='black')
    # plt.plot(x, m*(x) + b,color='black')
    h_indx = np.argsort(np.apply_along_axis(hor_comparator, 1, h))
    new_h = h[h_indx]
    func = interp1d(new_h[:,0],new_h[:,1],fill_value="extrapolate")
    x_new = np.arange(new_h[0][0],new_h[-1][0],0.1)
    y = func(x_new)
    non_nan_indices = np.argwhere(~np.isnan(y)).flatten()
    x_new = x_new[non_nan_indices]
    y = y[non_nan_indices]
    plt.plot(x_new,y,color='black')
    plt.imshow(img)
    return np.column_stack((x_new,y))

def lip_drawing(upper_lip, lower_lip, img):
    traj1,traj2 = [],[]
    idxes1 = concave_hull_indexes(upper_lip,length_threshold=5,)
    check = True
    for f, t in zip(idxes1[:-1], idxes1[1:]):  # noqa
        seg = upper_lip[[f, t]]
        if(check):
            check = False
            first = np.array(seg[0])
        traj1.append(seg[0])
        plt.plot(seg[:, 0], seg[:, 1],"-",color="black", alpha=0.5)
    traj1.append(first)
    check = True
    idxes2 = concave_hull_indexes(lower_lip,length_threshold=5,)
    for f, t in zip(idxes2[:-1], idxes2[1:]):  # noqa
        seg = lower_lip[[f, t]]
        if(check):
            check = False
            first2 = np.array(seg[0])
        traj2.append(seg[0])
        plt.plot(seg[:, 0], seg[:, 1],"-",color="black", alpha=0.5)
    traj2.append(first2)
    # for simplex in up_hull.simplices:

    #     plt.plot(upper_lip[simplex, 0], upper_lip[simplex, 1], color='black')
    # for simplex in down_hull.simplices:
    #     plt.plot(lower_lip[simplex, 0], lower_lip[simplex, 1], color='black')
    plt.imshow(img)
    return np.array(traj1), np.array(traj2)

def face_drawing(face,img):
    idxes = concave_hull_indexes(face,length_threshold=2,)
    traj = []
    check = True
    for f, t in zip(idxes[:-1], idxes[1:]):  # noqa
        seg = face[[f, t]]
        if(check):
            check = False
            first = np.array(seg[0])
        traj.append(seg[0])
        plt.plot(seg[:, 0], seg[:, 1],"-",color='black', alpha=0.5)
    traj.append(first)
    # indices = np.argsort(np.apply_along_axis(angle_with_x_axis, 1, face,center))
    # # angle = np.apply_along_axis(angle_with_x_axis, 1, face,center)
    # # print(angle)
    # # radius = -np.linalg.norm(face - center, axis=1)
    # # tie_break = np.lexsort((angle,radius))
    # face_border = face[indices]
    # for i in range(1,len(face_border)):
    #     plt.plot([face_border[i-1][0],face_border[i][0]],[face_border[i-1][1],face_border[i][1]],"-k")
    plt.imshow(img)
    return np.array(traj)


def hair_drawing(hair,img):
    # hull = ConvexHull(hair)
    # vertices = hair[hull.vertices]
    # h_idx = np.argsort(np.apply_along_axis(hor_comparator, 1, vertices))
    # v_idx = np.argsort(np.apply_along_axis(ver_comparator, 1, vertices))
    # horizontal_v = vertices[h_idx]
    # vertical_v = vertices[v_idx]
    # top = vertical_v[0]
    # left,right = [],[]
    # for pt in horizontal_v:
    #     if(pt[0] < top[0]):
    #         left.append(pt)
    #     elif(pt[0] > top[0]):
    #         right.append(pt)
    # left=np.array(left); right = np.array(right)
    # left_bottom = left[np.argmax(left[:,1])]
    # right_bottom = right[np.argmax(right[:,1])]
    # traj = []
    # first_x =  hair[hull.simplices[0], 0]; first_y = hair[hull.simplices[0], 1]
    # traj.append(np.array([first_x[1], first_y[1]]))
    # for simplex in hull.simplices:
    #     x_simplex = hair[simplex, 0]; y_simplex = hair[simplex, 1]
    #     if(x_simplex[0] == right_bottom[0] and x_simplex[1] == left_bottom[0] and y_simplex[0] == right_bottom[1] and y_simplex[1] == left_bottom[1]):
    #         print("check1")
    #         continue
    #     elif(x_simplex[0] == left_bottom[0] and x_simplex[1] == right_bottom[0] and y_simplex[0] == left_bottom[1] and y_simplex[1] == right_bottom[1]):
    #         print("check2")
    #         continue
    #     traj.append(np.array([x_simplex[1], y_simplex[1]]))
    #     plt.plot(x_simplex, y_simplex, "-k")
    idxes = concave_hull_indexes(hair,length_threshold=2,)
    traj = []
    check = True
    for f, t in zip(idxes[:-1], idxes[1:]): 
        seg = hair[[f, t]]
        if(check):
            check = False
            first = np.array(seg[0])
        traj.append(seg[0])
        plt.plot(seg[:, 0], seg[:, 1],"-",color='black', alpha=0.5)
    traj.append(first)
    plt.imshow(img)
    return np.array(traj)

data = np.array([1, 2, np.nan, 4, np.nan, 6])

# Find indices of non-NaN v

faceseg = FaceSegmentation()
# anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')
# img=cv2.imread('imgs/'+"test_logo"+'_resized.png')
# img = cv2.imread("WIN_20241210_10_37_43_Pro.jpg")
FACE_PORTRAIT = True
TEMP_DATA_DIR = 'temp_data/'

if FACE_PORTRAIT:
    # gray_image_masked,image_mask,face_mask,_ = faceseg.get_face_mask(img)
    # anime_img = anime.forward(gray_image_masked)
    # img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(TEMP_DATA_DIR+'img_out.jpg',anime_img)
    # edited_image = cv2.imread(TEMP_DATA_DIR+'img_out.jpg')
    ###############################################
    path = "C:\\Users\\Peijing Xu\\projects\\wen_research\\robotic_portrait\\Screenshot 2024-12-09 161747.png"
    # img = Image.open(path)
    # img = img.resize((300,500))
    # img.save(path)
    face_positions,vimg = faceseg.get_face_parts(path)
    flat_array = np.hstack(vimg.ravel())
    unique_numbers = np.unique(flat_array)
    print(unique_numbers)
    plt.imshow(vimg[0])
    plt.show()
    m,n = vimg[0].shape
    print(m,n)
    blank_image = Image.new('RGB',(n,m), 'white')
    f = []
    face_border = []
    right_brow = []
    left_brow = []
    right_eye = []
    left_eye = []
    nose = []
    upper_lip = []
    lower_lip = []
    hair = []
    white_color = np.array([255, 255, 255])  # White color in BGR format
    for face in vimg:
        for i in range(len(face)):
            for j in range(len(face[i])):
                if(face[i][j] == 23.181818):
                    f.append(np.array([j,i]))
                    if(i == len(face)-1 or j == len(face[i])-1 or i == 0 or j == 0):
                        face_border.append(np.array([j,i]))
                    elif(face[i+1][j] == 231.81819 or face[i-1][j] == 231.81819 
                                        or face[i][j+1] == 231.81819 or face[i][j-1] == 231.81819):
                        face_border.append(np.array([j,i]))
                    elif(face[i+1][j] == 0. or face[i-1][j] == 0. 
                                        or face[i][j+1] == 0. or face[i][j-1] == 0.):
                        face_border.append(np.array([j,i]))
                ############################################
                elif(face[i][j] == 46.363636):
                    right_brow.append(np.array([j,i]))
                    
                elif(face[i][j] == 69.545456):
                    left_brow.append(np.array([j,i]))

                elif(face[i][j] == 92.72727):
                    right_eye.append(np.array([j,i]))

                elif(face[i][j] == 115.909096):
                    left_eye.append(np.array([j,i]))

                elif(face[i][j] == 139.09091):
                    nose.append(np.array([j,i]))
                #############################################
                elif(face[i][j] == 162.27272):
                    upper_lip.append(np.array([j,i]))

                elif(face[i][j] == 208.63637):
                    lower_lip.append(np.array([j,i]))

                elif(face[i][j] == 231.81819):
                    hair.append(np.array([j,i]))
                ############################################

    f = np.array(f)
    face_border = np.array(face_border)
    right_brow = np.array(right_brow)
    left_brow = np.array(left_brow)
    right_eye = np.array(right_eye)
    left_eye = np.array(left_eye)
    nose = np.array(nose)
    upper_lip = np.array(upper_lip)
    lower_lip = np.array(lower_lip)
    hair = np.array(hair)
    hair_traj = hair_drawing(hair,blank_image)

    n_traj = nose_drawing(nose,face_positions[0][2],blank_image)
    if(len(right_eye) != 0 or len(right_brow) != 0):
        t1,t2,t3,t4 = eye_drawing(right_eye,right_brow,blank_image,position="right")
    if(len(left_eye) != 0 or len(left_brow) != 0):
        t5,t6,t7,t8 = eye_drawing(left_eye,left_brow,blank_image,position='left')
    up_traj, low_traj = lip_drawing(upper_lip,lower_lip,blank_image)
    # hair_traj = hair_drawing(hair,blank_image)
    face_traj = face_drawing(face_border,blank_image)

    np.savetxt("face_traj.csv",face_traj,delimiter=",")
    np.savetxt("hair_traj.csv",hair_traj,delimiter=",")
    np.savetxt("upp_traj.csv",up_traj,delimiter=",")
    np.savetxt("low_traj.csv",low_traj,delimiter=",")
    np.savetxt("up_r_eye.csv",t3,delimiter=",")
    np.savetxt("do_r_eye.csv",t4,delimiter=",")
    np.savetxt("up_l_eye.csv",t7,delimiter=",")
    np.savetxt("do_l_eye.csv",t8,delimiter=",")
    np.savetxt("r_brow1.csv",t1,delimiter=",")
    np.savetxt("r_brow2.csv",t2,delimiter=",")
    np.savetxt("l_brow1.csv",t5,delimiter=",")
    np.savetxt("l_brow2.csv",t6,delimiter=",")
    np.savetxt("nose_traj.csv",n_traj,delimiter=",")
    # vertices = points[hull.vertices]
    # indices = np.argsort(np.apply_along_axis(hor_comparator, 1, vertices))
    # hull_points = vertices[indices]
    # max_x = hull_points[-1][0]
    # i = len(hull_points)-1
    # start = hull_points[-1]
    # while(i >= 0):
    #     if(hull_points[i][0] == max_x and start[1] > hull_points[i][1]):
    #         start = hull_points[i]
    #     elif(hull_points[i][0] < max_x):
    #         break
    #     i -= 1
    # plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    #####################################################################
    
else:
    anime_img = cv2.imread(TEMP_DATA_DIR+'img_out.jpg')
    img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)




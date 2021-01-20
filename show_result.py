import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask_to_rgb(mask,rgb,color=np.array([142,0,0]),add_color=np.array([100,0,0])):
    """
    Add a mask on object in RGB object
        mask: segmented mask (numpy array)
        rbg: RGB image (numpy array)
        color: color pre-assigned to a class
        add_color: color want to add to object
    """
    x_indices, y_indices = np.where(np.all(mask == color, axis=-1))
    masked_rgb = rgb.copy()
    for i,j in zip(x_indices,y_indices):
        masked_rgb[i,j,:] = masked_rgb[i,j,:] + add_color
    return masked_rgb

def filter_contours(contours,boxes,percent=15):
    '''    
    Filter out contours < percent(%) and turn contours into hull
    '''
    areas = []
    new_contours = []
    new_boxes = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        areas.append(cv2.contourArea(hull))
    areas = np.array(areas)
    threshold = np.percentile(areas, percent)
    indexs = (areas>=threshold)
    indexs = np.where(indexs==True)[0]
    for index in indexs:
        new_contours.append(contours[index])
        new_boxes.append(boxes[index])
    return new_contours,new_boxes

def to_one_class_mask(mask,color=np.array([142,0,0])):
    x_indices, y_indices = np.where(np.all(mask == color, axis=-1))
    one_class_mask = np.zeros(mask.shape)
    for i,j in zip(x_indices,y_indices):
        one_class_mask[i,j,:] = one_class_mask[i,j,:] + color
    return one_class_mask

def find_contours(mask):
    h,w = mask.shape[:2]
    img = mask.copy()
    img = np.uint8(img)
    ret, thresh = cv2.threshold(img, 100, 255, 0)
    edged = cv2.Canny(thresh, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
        cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE) 
    return contours

def find_boxes(contours):
    boxes = []
    for contour in contours:
        boxes.append(cv2.boundingRect(contour)) # x,y,w,h
    return boxes

def find_objects(mask):
    contours = find_contours(mask)
    boxes = find_boxes(contours)
    contours, boxes = filter_contours(contours,boxes)
    objects = []
    h,w = mask.shape[:2]
    
    for contour,box in zip(contours,boxes):
        object_ = []
#         hull = cv2.convexHull(contour)
        hull = contour
        x,y,w,h = box
        for i in range(w):
            for j in range(h):
                if cv2.pointPolygonTest(hull, (x+i,y+j), False)!=-1:
                    object_.append((x+i,y+j))
        objects.append(object_)
    return objects

def find_distances(objects,depth,method='mean'):
    distances = []
    for object_ in objects:
        if method == 'mean':
            n = len(object_)
            mean_dist = 0
            for y,x in object_:
                mean_dist += depth[x,y]
            mean_dist = mean_dist/n
            distances.append(mean_dist)
        elif method == 'min':
            min_dist = 100
            for y,x in object_:
                if depth[x,y]<min_dist:
                    min_dist = depth[x,y]
            distances.append(min_dist)
    return distances

def show_distances(objects,rgb,distances,mask):
    locations = []
    for object_ in objects:
        object_ = np.array(object_)
        locations.append((int(np.mean(object_[:,0])),int(np.mean(object_[:,1]))))
    new_rgb = mask_to_rgb(mask,rgb)

    # Using cv2.putText() method 
    for loc,dist in zip(locations,distances):
        new_rgb = cv2.putText(new_rgb, str(np.round(dist,2)), loc, cv2.FONT_HERSHEY_SIMPLEX ,  
                       1, (0, 255, 0) , 2, cv2.LINE_AA) 
    return new_rgb

def to_rgb_with_distances(rgb,mask,depth):
    h,w = depth.shape
    rgb = cv2.resize(rgb,(w,h))
    mask = cv2.resize(mask,(w,h))
    one_class_mask = to_one_class_mask(mask)
    objects = find_objects(one_class_mask)
    distances = find_distances(objects,depth)
    rgb_with_distances = show_distances(objects,rgb,distances,one_class_mask)
    return rgb_with_distances

if __name__=='__main__':
    pass
    rgb = cv2.imread('rgb.png')
    mask = cv2.imread('mask.png')
    depth = np.load('depth.npy')
    cv2.imshow(to_rgb_with_distances(rgb,mask,depth))
    cv2.waitKey(0)

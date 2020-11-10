import cv2 as cv
points=[]
def lines(event,x,y,flags,params):
    if event == cv.EVENT_FLAG_LBUTTON:
        points.append((x,y))
        if len(points)%4 == 0:
            return


def draw2lines(img):
    cv.imshow("img", img)
    cv.setMouseCallback('img', lines)
    cv.waitKey(0)
    #print("from draw2lines",points)
    cv.destroyAllWindows()

def polygon_end_pts(l,height,width):
    t1,t2,t3,t4=l[0],l[1],l[2],l[3]
    x1,y1=t1[0],t1[1]
    x2,y2=t2[0],t2[1]
    m1 = (y2 - y1) / (x2 - x1)
    e1 = y1-(m1*x1)
    e2= y2+m1*(width-x2)
    x3,y3 = t3[0] , t3[1]
    x4,y4 = t4[0] , t4[1]
    m2 = (y4 - y3) / (x4 - x3)
    e3 = y3-(m2*x3)
    e4= y4+m2*(width-x4)
    return int(e1),int(e2),int(e3),int(e4)


def create_polygon(img):
    height, width = img.shape[:2]
    e1,e2,e3,e4 = polygon_end_pts(points,height,width)
    p1,p2,p3,p4=(0,e1),(width,e2),(0,e3),(width,e4 )
    cv.line(img, points[0], points[1], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img, points[2], points[3], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img,p1, points[0], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img, p2, points[1], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img,p3, points[2], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img, p4, points[3], (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img,p2, p4, (255, 0, 0), 5, cv.LINE_AA)
    cv.line(img, p1, p3, (255, 0, 0), 5, cv.LINE_AA)
    cv.imshow('poly',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return [p1,p2,p3,p4]


def calibrate(img):
    draw2lines(img)
    poly = create_polygon(img)
    points.clear()
    return poly







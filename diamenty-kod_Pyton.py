from exceptions import TypeError
import numpy as np
import cv2 as cv
from sympy import *
from sympy.geometry import *
from sympy import Point
from operator import itemgetter
import sys

input_image_path = None
output_image_path = None
if len(sys.argv) > 1:
    input_image_path = sys.argv[1]
if input_image_path is None:
    input_image_path = "jewels/Photo002.jpg"



class Data(object):
    NONE = 0
    SEQUENCE = 1
    IMAGE = 2

    def __init__(self, value=None, _type=IMAGE):
        self.value = value
        self._type = _type

    def type(self):
        if self.value is None or (self._type == Data.IMAGE and not self.value.size):
            return Data.NONE
        else:
            return self._type

    def sequence_get_value(self, sequence_number):
        if self.type() != Data.SEQUENCE:
            return self
        else:
            return self.value[min(sequence_number, len(self.value) - 1)]

    def desequence_all(self):
        """Zwraca tablice jednowymiarowa z wszystkimi wartosciami sekwencji"""
        if self.type() == Data.NONE: return []
        if self.type() == Data.IMAGE: return [self.value]
        if self.type() == Data.SEQUENCE:
            t = []
            for d in self.value:
                t += d.desequence_all()
            return t
        raise TypeError("Wrong data type - cannot desequence")

    def to_string(self):
        if self.type() == Data.NONE: return 'None_v'
        if self.type() == Data.IMAGE: return 'image_v'
        if self.type() == Data.SEQUENCE:
            s = ""
            images_count = 0
            none_count = 0
            for d in self.value:
                if d.type() == Data.SEQUENCE:
                    s += d.to_string() + ", "
                if d.type() == Data.IMAGE:
                    images_count += 1
                if d.type == Data.NONE:
                    none_count += 1
            if images_count:
                s += str(images_count) + " images, "
            if none_count:
                s += str(none_count) + " nones, "
            if len(s) > 0:
                s = s.rstrip(', ')
            s = 'Sequence_v' + '[' + s + ']'
            return s
        raise TypeError("Wrong data type - cannot desequence")

    def __nonzero__(self):
        return self.type() != Data.NONE

    def __eq__(self, other):
        #not tested!
        if not isinstance(other, Data): return False
        if self is not other: return False #???
        if self.type() != other.type(): return False
        if self.value is not other.value: return False #???
        if self.type() == Data.NONE:
            return True
        elif self.type() == Data.SEQUENCE:
            for a, b in zip(self.value, other.value):
                if a != b: return False
            print "eq=true:", self.value, other.value
            return True
        elif self.type() == Data.IMAGE:
            #assert isinstance(self.value, np.ndarray)
            return self.value is other.value and np.array_equal(self.value, other.value)
        else:
            raise ValueError("Wrong data type")










# inner code for codeelementex_9226526
def codeelementex_9226526_fun(in1, in2, in3, in4, parameters, memory): 
    (x0, y0, radius) = in2[0][0]
    mask = np.zeros(in1.shape, np.uint8)
    cv.circle(mask, (x0, y0), int(radius * 0.9), 1, -1)
    res = in1 * mask
    return res, mask * 255, None, None

# memory for codeelementex_9226526
codeelementex_9226526_memory = {}

# general code for codeelementex_9226526
def codeelementex_9226526(inputs, outputs, parameters):
    ins = [None] * 4
    for i in xrange(4):
        n = "in" + str(i + 1)
        if n in inputs and inputs[n]:
            ins[i] = inputs[n].value
    o = codeelementex_9226526_fun(ins[0], ins[1], ins[2], ins[3], parameters, codeelementex_9226526_memory)
    for i, v in enumerate(o):
        n = "o" + str(i + 1)
        outputs[n] = Data(v)

def opencvauto_houghlinesp( inputs, outputs, parameters):
    image = inputs['image'].value
    rho = parameters['rho']
    theta = parameters['theta']
    threshold = parameters['threshold']
    minLineLength = parameters['minLineLength']
    maxLineGap = parameters['maxLineGap']
    lines = cv.HoughLinesP(image=image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength,
                                        maxLineGap=maxLineGap)
    outputs['lines'] = Data(lines)


# inner code for codeelementex_22576926
def codeelementex_22576926_fun(in1, in2, in3, in4, parameters, memory):
    
    orig = in1
    nppoints = in2
    points = [Point(int(point[0]), int(point[1])) for point in nppoints]
    segments = [Point(int(point[0]), int(point[1])) for point in in3]
    npcenter = in4[0,0,:2]
    outer_diameter = in4[0,0,2]
    center = Point(npcenter)
    
    #podglad kolek
    prv = orig + 0
    #for c in points:
    #    cv.circle(prv, (c.x, c.y), 3, (0,255,0))
    #for s in segments:
    #    cv.circle(prv, (s.x, s.y), 3, (0,0,255))
    
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho
    
    def srt(p):
        pp = p - npcenter
        return cart2pol(pp[0],pp[1])[0]
    
    def calc_edgelen(centers):
        #obliczamy przyblizona dlugosc boku osmiokata
        centers_sorted = sorted(centers, key=srt)
        edgelens = [cv.norm(a-b) for a, b in zip(centers_sorted,centers_sorted[1:]+[centers_sorted[0]])]
        edgelen = np.mean(edgelens)
        if not all(0.8*edgelen < e < 1.25*edgelen for e in edgelens):
            print "WARNING: edge lenghts are not consistent!"
        return edgelen
        
    
    candidates = nppoints.astype(np.float32) + 0.0
    for maxerr in [0.5,0.25,0.1]:
        
    
        #obliczamy zgrubsza wierzcholki osmiokata
        _, labels, centers = cv.kmeans(candidates, 8, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_PP_CENTERS)
    
        edgelen = calc_edgelen(centers)
    
        #odsiewamy punkty oddalone o wiecej miz maxerr
        candidates2 = []
        labels2 = []
        for point, label in zip(candidates, labels):
            center = centers[label]
            if cv.norm(point-center) < maxerr*edgelen:
                candidates2.append(point)
                labels2.append(label)
        candidates = np.array(candidates2)
        labels = np.array(labels2)
    
    #for c in candidates:
    #    cv.circle(prv, (c[0],c[1]), 3, (0,0,255), -1)
    
    #jeszcze raz selekcjonujemy kandydatow
    for i in xrange(5):
        centers2 = []
        for label in xrange(8):
            cands = candidates[(labels==label)[...,0]]
            cent = centers[label]
            cents = [c for c in centers if 0.7*edgelen<cv.norm(c-cent)<1.4*edgelen]
            if len(cents) != 2:
                print "WARNING: wrong number of nearest centers!"
                centers2.append(cent)
                continue
            A, B = cents
            distances = [abs(cv.norm(A-c)-edgelen)+abs(cv.norm(B-c)-edgelen) for c in cands]
            best = cands[np.argmin(distances)]
            centers2.append(best)
        centers = np.array(centers2,np.float32)
        edgelen = calc_edgelen(centers)
                    
    for c in centers:
        cv.circle(prv, (c[0],c[1]), 9, (0,255,255), -1)
    cv.circle(prv, (npcenter[0], npcenter[1]), outer_diameter, (0,255,255), 6)
    
    center2 = np.mean(centers, axis=0)
    #cv.circle(prv, (npcenter[0],npcenter[1]), 3, (255,80,80), -1)
    #cv.circle(prv, (center2[0],center2[1]), 3, (0,255,255), -1)
    
    inner_diameter = np.mean([cv.norm(center2-c) for c in centers])
    
    
    
    o = np.zeros((70,300),np.uint8)
    cv.putText(o, "Outer diameter: " + str(outer_diameter*2), (10, 20), 2, 0.5, 255, 1, cv.CV_AA)
    cv.putText(o, "Inner diameter: " + str(inner_diameter*2), (10, 40), 2, 0.5, 255, 1, cv.CV_AA)
    cv.putText(o, "     Proportion: " + str(inner_diameter/outer_diameter), (10, 60), 2, 0.5, 255, 1, cv.CV_AA)
    
    print inner_diameter/outer_diameter
    
    return prv, o

# memory for codeelementex_22576926
codeelementex_22576926_memory = {}

# general code for codeelementex_22576926
def codeelementex_22576926(inputs, outputs, parameters):
    ins = [None] * 4
    for i in xrange(4):
        n = "in" + str(i + 1)
        if n in inputs and inputs[n]:
            ins[i] = inputs[n].value
    o = codeelementex_22576926_fun(ins[0], ins[1], ins[2], ins[3], parameters, codeelementex_22576926_memory)
    for i, v in enumerate(o):
        n = "o" + str(i + 1)
        outputs[n] = Data(v)

def opencvauto_houghcircles( inputs, outputs, parameters):
    image = inputs['image'].value
    method = parameters['method']
    dp = parameters['dp']
    minDist = parameters['minDist']
    param1 = parameters['param1']
    param2 = parameters['param2']
    minRadius = parameters['minRadius']
    maxRadius = parameters['maxRadius']
    circles = cv.HoughCircles(image=image, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    outputs['circles'] = Data(circles)


# inner code for codeelementex_22574838
def codeelementex_22574838_fun(in1, in2, in3, in4, parameters, memory):
    out1 = in1.copy()
    out2 = in1.copy()
    lines = in2
    circles = in3
    
    #
    # Output 1
    #
    
    (x0, y0, radius) = circles[0][0]
    p0 = Point(x0, y0)
    cv.circle(out1, (x0, y0), radius, (0, 255, 0), 2)
    cv.circle(out1, (x0, y0), 2, (0, 0, 255), 3)
    
    
    segments=[]
    
    for x1,y1,x2,y2 in lines[0]:
        
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        line = Line(p1, p2)
    
        perpendicular = line.perpendicular_line(p0)
        pot_side_center = line.intersection(perpendicular)[0]
        pot_radius = float(p0.distance(pot_side_center))
        pot_side = pot_radius * 0.87 # stosunek promienia okregu opisanego do boku osmiokata
        
        a =    pot_side * 0.5
        segment = Segment(p1, p2)
        if pot_radius < radius * 0.35 or (not segment.contains(pot_side_center) and (p1.distance(pot_side_center) > a and p2.distance(pot_side_center) > a)):
            continue
    
        cv.line(out1,(x1,y1),(x2,y2),(0,255,0),2)
        cv.circle(out1, (pot_side_center.x, pot_side_center.y), 2, (0, 255, 255), 3)
    
    #
    # Output 2
    #
    
        circle = Circle(pot_side_center, a * 1.5)
        intersection = line.intersection(circle)
        inter1 = intersection[0]
        inter2 = intersection[1]
        inter1 = Point(int(inter1.x), int(inter1.y))
        inter2 = Point(int(inter2.x), int(inter2.y))
        cv.line(out2,(inter1.x, inter1.y),(inter2.x, inter2.y),(0,255,0),2)
        segments.append(Segment(inter1, inter2))
    
    #
    # Output 4
    #
    
    points=[]
    for segmentA in segments:
        
        for segmentB in segments:
            if segmentA is not segmentB:
                intersection = segmentA.intersection(segmentB)
                if len(intersection) > 0:
                    if isinstance(intersection[0], Point):
                        point = Point(int(intersection[0].x), int(intersection[0].y))
                        points.append(point)
    
    nppoints = np.zeros((len(points), 2), np.float)
    i = 0
    for point in points:
        nppoints[i] = (point.x, point.y)
        i += 1
    
    npsegments = np.zeros((len(points), 2), np.float)
    i = 0
    for segment in segments:
        npsegments[i] = (segment.midpoint.x, segment.midpoint.y)
        i += 1
    
    return out1, out2, npsegments, nppoints

# memory for codeelementex_22574838
codeelementex_22574838_memory = {}

# general code for codeelementex_22574838
def codeelementex_22574838(inputs, outputs, parameters):
    ins = [None] * 4
    for i in xrange(4):
        n = "in" + str(i + 1)
        if n in inputs and inputs[n]:
            ins[i] = inputs[n].value
    o = codeelementex_22574838_fun(ins[0], ins[1], ins[2], ins[3], parameters, codeelementex_22574838_memory)
    for i, v in enumerate(o):
        n = "o" + str(i + 1)
        outputs[n] = Data(v)

def opencvcanny( inputs, outputs, parameters):
    image = inputs["input"].value

    color = False
    if len(image.shape) >= 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        color = True

    output = cv.Canny(image, parameters["thr1"], parameters["thr2"])

    if color:
        output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

    outputs["output"] = Data(output)

def imageloader( inputs, outputs, parameters):
    d = cv.imread(parameters["path"])
    if d is not None:
        
        outputs["output"] = Data(d)

def colorconverter( inputs, outputs, parameters):
    if parameters["code"] is None:
        outputs["output"] = Data(inputs["input"].value.copy())
    else:
        outputs["output"] = Data(cv.cvtColor(inputs["input"].value, parameters["code"]))

def opencvmorphologyex( inputs, outputs, parameters):
    image = np.copy(inputs["input"].value)
    operation = parameters["operation"]
    element_type = parameters["element type"]
    element_size = parameters["element size"]
    iterations = parameters["iterations"]

    element = cv.getStructuringElement(element_type, (element_size, element_size))
    image = cv.morphologyEx(image, operation, element, iterations=iterations,)
    outputs["output"] = Data(image)

imageloader2 = {}
imageloader({}, imageloader2, {"path":input_image_path})
colorconverter7 = {}
colorconverter({"input":imageloader2["output"]}, colorconverter7, {"code":6L})
opencvcanny6 = {}
opencvcanny({"input":colorconverter7["output"]}, opencvcanny6, {"thr1":19,"thr2":37})
opencvmorphologyex9 = {}
opencvmorphologyex({"input":colorconverter7["output"]}, opencvmorphologyex9, {"operation":2L,"element type":2L,"element size":9,"iterations":20})
opencvauto_houghcircles8 = {}
opencvauto_houghcircles({"image":opencvmorphologyex9["output"]}, opencvauto_houghcircles8, {"method":3,"dp":1.4,"minDist":1000.0,"param1":200.0,"param2":80.0,"minRadius":1,"maxRadius":0})
codeelementex5 = {}
codeelementex_9226526({"in1":opencvcanny6["output"],"in2":opencvauto_houghcircles8["circles"]}, codeelementex5, {"code":u'import cv2 as cv\nimport numpy as np\nfrom sympy import *\nfrom sympy.geometry import *\n\n\n(x0, y0, radius) = in2[0][0]\nmask = np.zeros(in1.shape, np.uint8)\ncv.circle(mask, (x0, y0), int(radius * 0.9), 1, -1)\nres = in1 * mask\nreturn res, mask * 255, None, None',"split_channels":False})
opencvauto_houghlinesp4 = {}
opencvauto_houghlinesp({"image":codeelementex5["o1"]}, opencvauto_houghlinesp4, {"rho":1.0,"theta":0.01,"threshold":40,"minLineLength":40.0,"maxLineGap":15.0})
codeelementex3 = {}
codeelementex_22574838({"in1":imageloader2["output"],"in2":opencvauto_houghlinesp4["lines"],"in3":opencvauto_houghcircles8["circles"]}, codeelementex3, {"code":u'import cv2 as cv\nimport numpy as np\nfrom sympy import *\nfrom sympy.geometry import *\n\n\n\nout1 = in1.copy()\nout2 = in1.copy()\nlines = in2\ncircles = in3\n\n#\n# Output 1\n#\n\n(x0, y0, radius) = circles[0][0]\np0 = Point(x0, y0)\ncv.circle(out1, (x0, y0), radius, (0, 255, 0), 2)\ncv.circle(out1, (x0, y0), 2, (0, 0, 255), 3)\n\n\nsegments=[]\n\nfor x1,y1,x2,y2 in lines[0]:\n\tintpoint()\n\tp1 = Point(x1, y1)\n\tp2 = Point(x2, y2)\n\tline = Line(p1, p2)\n\n\tperpendicular = line.perpendicular_line(p0)\n\tpot_side_center = line.intersection(perpendicular)[0]\n\tpot_radius = float(p0.distance(pot_side_center))\n\tpot_side = pot_radius * 0.87 # stosunek promienia okregu opisanego do boku osmiokata\n\t\n\ta =\tpot_side * 0.5\n\tsegment = Segment(p1, p2)\n\tif pot_radius < radius * 0.35 or (not segment.contains(pot_side_center) and (p1.distance(pot_side_center) > a and p2.distance(pot_side_center) > a)):\n\t\tcontinue\n\n\tcv.line(out1,(x1,y1),(x2,y2),(0,255,0),2)\n\tcv.circle(out1, (pot_side_center.x, pot_side_center.y), 2, (0, 255, 255), 3)\n\n#\n# Output 2\n#\n\n\tcircle = Circle(pot_side_center, a * 1.5)\n\tintersection = line.intersection(circle)\n\tinter1 = intersection[0]\n\tinter2 = intersection[1]\n\tinter1 = Point(int(inter1.x), int(inter1.y))\n\tinter2 = Point(int(inter2.x), int(inter2.y))\n\tcv.line(out2,(inter1.x, inter1.y),(inter2.x, inter2.y),(0,255,0),2)\n\tsegments.append(Segment(inter1, inter2))\n\n#\n# Output 4\n#\n\npoints=[]\nfor segmentA in segments:\n\tintpoint()\n\tfor segmentB in segments:\n\t\tif segmentA is not segmentB:\n\t\t\tintersection = segmentA.intersection(segmentB)\n\t\t\tif len(intersection) > 0:\n\t\t\t\tif isinstance(intersection[0], Point):\n\t\t\t\t\tpoint = Point(int(intersection[0].x), int(intersection[0].y))\n\t\t\t\t\tpoints.append(point)\n\nnppoints = np.zeros((len(points), 2), np.float)\ni = 0\nfor point in points:\n\tnppoints[i] = (point.x, point.y)\n\ti += 1\n\nnpsegments = np.zeros((len(points), 2), np.float)\ni = 0\nfor segment in segments:\n\tnpsegments[i] = (segment.midpoint.x, segment.midpoint.y)\n\ti += 1\n\nreturn out1, out2, npsegments, nppoints',"split_channels":False})
codeelementex1 = {}
codeelementex_22576926({"in4":opencvauto_houghcircles8["circles"],"in1":imageloader2["output"],"in2":codeelementex3["o4"],"in3":codeelementex3["o3"]}, codeelementex1, {"code":u'import cv2 as cv\nimport numpy as np\nfrom sympy import Point\nfrom operator import itemgetter\n\nprint "--------------"\n\norig = in1\nnppoints = in2\npoints = [Point(int(point[0]), int(point[1])) for point in nppoints]\nsegments = [Point(int(point[0]), int(point[1])) for point in in3]\nnpcenter = in4[0,0,:2]\nouter_diameter = in4[0,0,2]\ncenter = Point(npcenter)\n\n#podglad kolek\nprv = orig + 0\n#for c in points:\n#\tcv.circle(prv, (c.x, c.y), 3, (0,255,0))\n#for s in segments:\n#\tcv.circle(prv, (s.x, s.y), 3, (0,0,255))\n\ndef cart2pol(x, y):\n\ttheta = np.arctan2(y, x)\n\trho = np.hypot(x, y)\n\treturn theta, rho\n\ndef srt(p):\n\tpp = p - npcenter\n\treturn cart2pol(pp[0],pp[1])[0]\n\ndef calc_edgelen(centers):\n\t#obliczamy przyblizona dlugosc boku osmiokata\n\tcenters_sorted = sorted(centers, key=srt)\n\tedgelens = [cv.norm(a-b) for a, b in zip(centers_sorted,centers_sorted[1:]+[centers_sorted[0]])]\n\tedgelen = np.mean(edgelens)\n\tif not all(0.8*edgelen < e < 1.25*edgelen for e in edgelens):\n\t\tprint "WARNING: edge lenghts are not consistent!"\n\treturn edgelen\n\t\n\ncandidates = nppoints.astype(np.float32) + 0.0\nfor maxerr in [0.5,0.25,0.1]:\n\tintpoint()\n\n\t#obliczamy zgrubsza wierzcholki osmiokata\n\t_, labels, centers = cv.kmeans(candidates, 8, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_PP_CENTERS)\n\n\tedgelen = calc_edgelen(centers)\n\n\t#odsiewamy punkty oddalone o wiecej miz maxerr\n\tcandidates2 = []\n\tlabels2 = []\n\tfor point, label in zip(candidates, labels):\n\t\tcenter = centers[label]\n\t\tif cv.norm(point-center) < maxerr*edgelen:\n\t\t\tcandidates2.append(point)\n\t\t\tlabels2.append(label)\n\tcandidates = np.array(candidates2)\n\tlabels = np.array(labels2)\n\n#for c in candidates:\n#\tcv.circle(prv, (c[0],c[1]), 3, (0,0,255), -1)\n\n#jeszcze raz selekcjonujemy kandydatow\nfor i in xrange(5):\n\tcenters2 = []\n\tfor label in xrange(8):\n\t\tcands = candidates[(labels==label)[...,0]]\n\t\tcent = centers[label]\n\t\tcents = [c for c in centers if 0.7*edgelen<cv.norm(c-cent)<1.4*edgelen]\n\t\tif len(cents) != 2:\n\t\t\tprint "WARNING: wrong number of nearest centers!"\n\t\t\tcenters2.append(cent)\n\t\t\tcontinue\n\t\tA, B = cents\n\t\tdistances = [abs(cv.norm(A-c)-edgelen)+abs(cv.norm(B-c)-edgelen) for c in cands]\n\t\tbest = cands[np.argmin(distances)]\n\t\tcenters2.append(best)\n\tcenters = np.array(centers2,np.float32)\n\tedgelen = calc_edgelen(centers)\n\t\t\t\t\nfor c in centers:\n\tcv.circle(prv, (c[0],c[1]), 9, (0,255,255), -1)\ncv.circle(prv, (npcenter[0], npcenter[1]), outer_diameter, (0,255,255), 6)\n\ncenter2 = np.mean(centers, axis=0)\n#cv.circle(prv, (npcenter[0],npcenter[1]), 3, (255,80,80), -1)\n#cv.circle(prv, (center2[0],center2[1]), 3, (0,255,255), -1)\n\ninner_diameter = np.mean([cv.norm(center2-c) for c in centers])\n\n\n\no = np.zeros((70,300),np.uint8)\ncv.putText(o, "Outer diameter: " + str(outer_diameter*2), (10, 20), 2, 0.5, 255, 1, cv.CV_AA)\ncv.putText(o, "Inner diameter: " + str(inner_diameter*2), (10, 40), 2, 0.5, 255, 1, cv.CV_AA)\ncv.putText(o, "     Proportion: " + str(inner_diameter/outer_diameter), (10, 60), 2, 0.5, 255, 1, cv.CV_AA)\n\n\nreturn prv, o',"split_channels":False})

cv.imwrite("output1.png", codeelementex1["o1"].value)
cv.imwrite("output2.png", codeelementex1["o2"].value)
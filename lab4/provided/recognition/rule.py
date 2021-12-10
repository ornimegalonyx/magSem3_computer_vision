import numpy as np


class Rule:

    def __init__(self, areapoints):
        self.areapoints = np.array(areapoints)
        self.arealines = []
        self.setarealines(self.areapoints)

    def setarealines(self, areapoints):
        for curr in range(0, areapoints.shape[0]):
            next = curr + 1
            if next == areapoints.shape[0]:
                next = 0
            self.arealines.append(self.linebypoints(areapoints[curr], areapoints[next]))

    def linebypoints(self, point0, point1):
        k = (point0[1] - point1[1]) / (point0[0] - point1[0])
        b = point1[1] - k * point1[0]
        return [k, b]

    def comaprewithline(self, line, point):

        x = point[0]
        yp = point[1]
        k = line[0]
        b = line[1]
        yl = k * x + b

        return yp > yl

    def check(self, point):
        c = []
        for line in self.arealines:
            c.append(self.comaprewithline(line, point))

        return c
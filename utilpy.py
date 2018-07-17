import numpy

WIDTH = 2200
HEIGHT = 1400
CELL = 50

def parse(line, cell, width, height) :
    str_arr = line.split()
    cvar_str = str_arr[0].translate(None, '()').replace(',', ' ').rstrip()
    samples_str = str_arr[1].translate(None, '()').replace(',', ' ').rstrip()
    cvars = [float(s) for s in cvar_str.split(' ')]
    samples = [float(s) for s in samples_str.split(' ')]
    start = numpy.array(cvars[:4])
    goal = numpy.array(cvars[4:8])
    obs = numpy.reshape(cvars[8:],[9,2])
    grid = grid_map(obs, cell, width, height, cell)
    samples = numpy.reshape(samples, [len(samples)/4, 4])
    ret = [[obs, start, goal, samples[int(i)], numpy.concatenate(grid)] for i in numpy.linspace(0, len(samples)-1, 3)]
    return ret

def parse_sample_condition(line, cell, width, height) :
    value = parse(line, cell, width, height)
    samples = []
    conditions = [] 
    # print value
    for _, start, goal, s, occ in value :
        samples.append(s)
        conditions.append(numpy.concatenate((start,goal,occ)))
    return samples, conditions

def parse_samples_conditions(lines, indexes, cell, width, height) :
    samples = []
    conditions = [] 
    for i in indexes :
        value = parse(lines[i], cell, width, height)
        # print value
        for _, start, goal, s, occ in value :
            samples.append(s)
            conditions.append(numpy.concatenate((start,goal,occ)))
    return samples, conditions

def circle(pt, radius) :
    ret = []
    x = radius
    y = 0
    err = 0
    while x >= y :
        ret.append([pt[0]+x, pt[1]+y])
        ret.append([pt[0]+y, pt[1]+x])
        ret.append([pt[0]-y, pt[1]+x])
        ret.append([pt[0]-x, pt[1]+y])
        ret.append([pt[0]-x, pt[1]-y])
        ret.append([pt[0]-y, pt[1]-x])
        ret.append([pt[0]+y, pt[1]-x])
        ret.append([pt[0]+x, pt[1]-y])
        y += 1
        if err <= 0 : 
            err += 2*y+1
        else :
            x -= 1
            err += 2*(y-x) + 1
    for i in range(radius) :
        for j in range(radius) :
            ret.append([pt[0]+i, pt[1]+j])
            ret.append([pt[0]-i, pt[1]-j])
            ret.append([pt[0]+i, pt[1]-j])
            ret.append([pt[0]-i, pt[1]+j])
    return ret

def occupied_cells(obs, radius, cell_size) :
    obstacles = [[o[0]*100, o[1]*100.0] for o in obs]
    cs = cell_size
    circles = []
    for o in obstacles :
        pt = [int(o[0]/cs), int(o[1]/cs)]
        c = circle(pt, int(radius/cs))
        circles.extend(c)
    return circles

def grid_map(obs, radius, width, height, cell_size) :
    circles = occupied_cells(obs, radius, cell_size)
    w = int(width/cell_size)
    h = int(height/cell_size)
    grid = numpy.zeros((w,h))
    for c in circles :
        grid[min(c[0]+w/2,w-1),min(c[1]+h/2,h-1)] = 1.0
    return grid

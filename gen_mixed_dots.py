### Master Dataset Generator ###
'''

This dataset generator combines various dot image styles into 
a single generator. Every output image is named as follows;

[# of dots]_[image color style]_[image size style]_[unique image #].png

Here is a description of the color style types generated;

bow:
   Black dots on a white background
wob:
   White dots on a black background
greyrandom:
   grey dots on random grey background with a random grey color assigned to each grey dot
greysingle:
    grey dots on random grey background with a single random grey color assigned to all dots
colorrandom:
   color dots on random color background with a random color assigned to each color dot
colorsingle:
    color dots on random color background with a single random color assigned to all dots

here is a description of the size style types generated;

random:
    each dot generated is a random size within the range
dotareacontrol:
    a single small dot size is used for every dot generated 
totalareacontrolsame:
    every image uses the same total dot area, and for each class (number of dots) the size of every
    dot is the same
totalareacontroldifferent:
    every image uses the same total dot area, the sizes of inidivual dots within each image still vary
    as determined by gen_dot_sizes_same_area()
'''


from PIL import Image, ImageDraw
import random
import numpy as np
import os
import scipy.stats as ss
import time


### PARAMETERS (Set these) ###

pic_dim = 256     # pixel dimension of one axis of output image
max_radius = 30   # maximum dot size
min_radius = 2    # minimum dot size
dot_dist = 3      # minimum distance between the edge of two dots
color_dist = 40   # minimum RGB distance between different colors in the image, background or between dots
max_dots = 20   # maximum number of dots in the image
num_pics_per_catergory = 20  # number of pictures per category
min_dots = 1     # one is most natural here, can be zero, this will include blank backgrounds in your dataset (good to prevent edge effect) 

size_styles = ['totalareacontroldifferent', 'dotareacontrol','random']
color_styles = ['wob']

outputdir = '../../stimuli/enumeration_unit_selection/wob/unit_select_2'


### STOP EDITING ###



### FUNCTIONS ###

#Pick a number from a range of 0 to max integer, based on a gaussian distribution around the middle number
def gaussian_choice(num_range):
    x = np.arange(1,num_range)
    if num_range%2 == 0:
        #print(str(np.floor(num_range/2)+1-num_range)+' to '+str(np.floor(num_range/2)))
        y = np.arange(np.floor(num_range/2)+1-num_range, np.floor(num_range/2))
    else:
        #print(str(np.floor(num_range/2)+1-num_range) + ' to '+ str(np.floor(num_range/2)))
        y = np.arange(np.floor(num_range/2)+1-num_range, np.floor(num_range/2))
    yU, yL = y + 0.5, y - 0.5 
    prob = ss.norm.cdf(yU, scale = 2) - ss.norm.cdf(yL, scale = 2)
    prob = prob / prob.sum() #normalize the probabilities so their sum is 1
    #print(prob)
    return np.random.choice(x, p = prob)

def gen_radii_areacontroldiff(average_radius,num_dots, min_radius=min_radius,max_radius=max_radius):
    average_area = round(np.pi*average_radius**2,1)
    total_area = average_area*num_dots
    if num_dots == 1:
        return {1:average_radius}
    else:
        radii = {}
        num_below = gaussian_choice(num_dots)
        extra_area = 0
        for i in range(num_below):
            radii[num_dots-i] = round(np.random.uniform(min_radius,average_radius),1)
            extra_area += average_area - round(np.pi*radii[num_dots-i]**2,1)
        for i in range(1,num_dots-num_below+1):
            added_area = round(np.random.uniform(0,extra_area),1)
            radii[i] = round(np.sqrt((average_area+added_area)/np.pi),1)
            extra_area -= added_area
        return radii

def dot_size_position_generator(style, num_dots, pic_dim = pic_dim, max_dots = max_dots, dot_dist = dot_dist,
                                max_radius = max_radius, min_radius = min_radius):
    average_radius = round(np.sqrt(max_dots*(min_radius+3)**2/num_dots),1)
    retry = True
    while retry:
        retry = False
        sizes = {}

        if style == 'totalareacontroldifferent':
            tacd_radii = gen_radii_areacontroldiff(average_radius,num_dots)
            total_area = 0 
            for key in tacd_radii:
                total_area += np.pi*tacd_radii[key]**2
            #print(tacd_radii)
            #print('total_area: %s'%str(total_area))
    
        for i in range(1,num_dots+1):
            #get spatial position
            touching = True
            attempts = 0
            while touching:
                attempts += 1
                if style == 'random':
                    r = round(np.random.uniform(min_radius,max_radius),1)
                elif style == 'dotareacontrol':
                    r = min_radius + 2
                elif style == 'totalareacontrolsame':
                    r = average_radius
                elif style == 'totalareacontroldifferent':
                    r = tacd_radii[i]
                x = round(np.random.uniform(r,pic_dim-r),1)
                y = round(np.random.uniform(r,pic_dim-r),1)
                touching = False
                for dot in sizes:
                    distance = np.sqrt((x-sizes[dot][0])**2+(y-sizes[dot][1])**2)
                    if distance <= r+sizes[dot][2]+dot_dist:
                        if attempts >= 200:
                            retry = True
                            break
                        touching = True
                        break
            if retry:
                break
            sizes[i] = [x,y,r]      
    return sizes

def dot_color_generator(style,num_dots, color_dist = color_dist):
    colors = {}
    pythag_color_dist = np.sqrt(color_dist**2*3)

    if style == 'bow':
        colors['background'] = (255,255,255)
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (0,0,0)
        return colors

    elif style == 'wob':
        colors['background'] = (0,0,0)
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (255,255,255)
        return colors       

    elif style == 'greyrandom':
        background_color = random.randint(0,255)
        colors['background'] = (background_color,background_color,background_color)
        for dot_num in range(1,num_dots+1):
            camo = True
            while camo:
                c = random.randint(0,255)
                if abs(c-background_color) > color_dist:
                    camo = False
            colors[dot_num] = (c,c,c)

    elif style == 'greysingle':
        background_color = random.randint(0,255)
        colors['background'] = (background_color,background_color,background_color) 
        camo = True
        while camo:
            c = random.randint(0,255)
            if abs(c-background_color) > color_dist:
                camo = False
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = (c,c,c)

    elif style == 'colorrandom':
        colors['background'] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        for dot_num in range(1,num_dots+1):
            camo = True
            while camo:
                c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                if np.sqrt((c[0]-colors['background'][0])**2+(c[1]-colors['background'][1])**2+(c[2]-colors['background'][2])**2) > pythag_color_dist:
                    camo = False
            colors[dot_num] = c

    elif style =='colorsingle':
        colors['background'] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        camo = True
        while camo:
            c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            if np.sqrt((c[0]-colors['background'][0])**2+(c[1]-colors['background'][1])**2+(c[2]-colors['background'][2])**2) > pythag_color_dist:
                camo = False
        for dot_num in range(1,num_dots+1):
            colors[dot_num] = c
    return colors


### MAIN SCRIPT ###

#Setup path
if not os.path.exists(outputdir):
    os.mkdir(outputdir)
outputfile = open(os.path.join(outputdir,'img_stats.tsv'),'w+')


#Setup header for output image stats file
column_nums = range(1,max_dots+1)
column_names = []
for i in column_nums:
    column_names.append('dot'+str(i))
outputfile.write("image name    %s\n"%'    '.join(column_names))
outputfile.flush()
#Main Image Generation Loop
start_time = time.time()

image_index = 0

for num_dots in range(min_dots,max_dots+1):
    print(num_dots)
    for size_style in size_styles:
        if not os.path.exists(os.path.join(outputdir,size_style)):
            os.mkdir(os.path.join(outputdir,size_style))
        for color_style in color_styles:
            if not os.path.exists(os.path.join(outputdir,size_style,color_style)):
                os.mkdir(os.path.join(outputdir,size_style,color_style))
            fulloutputdir = os.path.join(outputdir,size_style,color_style)
            for pic in range(1,num_pics_per_catergory+1):

                image_index += 1
                #img_file_name = '%s_%s_%s_%s_%s.png'%(num_dots,color_style,size_style,pic,image_index)
                img_file_name = '%s_%s_%s_%s.png'%(num_dots,color_style,size_style,pic)
                toprint = img_file_name

                colors = dot_color_generator(color_style,num_dots)
                img = Image.new('RGB', (pic_dim, pic_dim), color = colors['background'])

                if num_dots > 0:
                    sizes = dot_size_position_generator(size_style,num_dots)
                    for dot_num in range(1,num_dots+1):
                        toprint += '    '+str(sizes[dot_num])+str(colors[dot_num])
                        corners = [sizes[dot_num][0]-sizes[dot_num][2],sizes[dot_num][1]-sizes[dot_num][2],sizes[dot_num][0]+sizes[dot_num][2],sizes[dot_num][1]+sizes[dot_num][2]]
                        dotdraw = ImageDraw.Draw(img)
                        dotdraw.ellipse(corners, fill=(colors[dot_num]), outline=(colors[dot_num]))
                
                img.save(os.path.join(fulloutputdir,img_file_name))
                for i in range(max_dots-num_dots):
                    toprint+='    '
                outputfile.write(toprint+'\n')
                outputfile.flush()

end_time = time.time()
print('Run Time: %s'%(end_time-start_time))


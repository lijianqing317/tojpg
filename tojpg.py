import PIL.Image as im
import os
import shutil
one_level_dictory =['Training','Testing']
for i in one_level_dictory:
    second_level_dictory = os.listdir(r'./%s'%i)
    for s in second_level_dictory:
        path = i+'/'+s
        save_path = './imageData/%s/%s'%(i,s)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for j in os.listdir(r'%s'%path):
            if j.endswith('ppm'):
                imageName,imageType = os.path.splitext(j)
                image = im.open(path+'/'+j)
                image.save(save_path+'/%s.jpg'%imageName)
                # shutil.copy(path+'/'+j,pp+'/'+j)


import PIL.Image as im
import os
import shutil
def resizeMethod(saveFileName,saveImageType):
    one_level_dictory =['Training','Testing']
    for i in one_level_dictory:
        second_level_dictory = os.listdir(r'./%s'%i)
        for s in second_level_dictory:
            path = i+'/'+s
            #save_path = './imageData/%s/%s'%(i,s)
            #if not os.path.exists(save_path):
                #os.makedirs(save_path)
            resize_save_path = './%s/%s/%s'%(saveFileName,i,s)
            if not os.path.exists(resize_save_path):
                os.makedirs(resize_save_path)
            for j in os.listdir(r'%s'%path):
                if j.endswith('ppm'):
                    imageName,imageType = os.path.splitext(j)
                    image = im.open(path+'/'+j)
                    out = image.resize((28,28),im.ANTIALIAS)
                    #image.save(save_path+'/%s.jpg'%imageName)
                    out.save(resize_save_path+'/%s%s'%(imageName,saveImageType))
                    # shutil.copy(path+'/'+j,pp+'/'+j)
resizeMethod('test','.png')


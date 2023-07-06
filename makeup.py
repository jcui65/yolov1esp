
#file = open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config')
'''
with open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('green', 'blue')
with open('file.txt', 'w') as file:
  file.write(filedata)
'''


import os, sys

path = '/home/jianning/model_editor_models/'#unit_box_0bluesmallsd311/'# "/var/www/html/"
dirs = os.listdir(path)#that meets my need!


for newshouldbe in ['smallt42','smallt43','smallt44']:#['t411','t412','t413','t421','t422','t423','t431','t432','t433','t441','t442','t443']:#['t411','t412','t413','t416','t419','t424',,'t434']:
    #oldreal = 'real'
    #newusedtobe = newshouldbe + oldreal
    for file in dirs:
        #shouldbe = 't434real'
        if newshouldbe in file:
            print('the right file found!')
            print(file)
            for j in ['/model.sdf']:#'/model.config',
                print('now working on: '+j)
                filein=path+file+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
                f = open(filein, 'r')
                filedata = f.read()
                f.close()
                file1 = file[:-3]
                file2 = file[-3:-2]
                file3 = file[-2:]
                file3int = int(file3)
                file3int2 = file3int +8  # for 20.0#6#6 for 5.0#
                sfile3 = str(file3int2)
                newdata2=filedata.replace("libanimated_boxt4","libabt4")
                #newdata2 = newdata.replace(newusedtobe, newshouldbe)
                #newdata = filedata.replace("0 1 0 1","0 0 1 1")#("green","blue")
                #newdata2=newdata.replace("green","blue")
                filedir=path+file#+j
                fileout= filedir+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                #if not os.path.isdir(fileout):
                    #os.makedirs(fileout)
                f = open(fileout,'w+')
                f.write(newdata2)
                f.close()


'''
for file in dirs:
    #
    #print('the folder is loaded!')
    usedtobe="t434"
    shouldbe='t434real'
    if shouldbe in file:
        print('the right file found!')
        print(file)
        for j in ['/model.config','/model.sdf']:
            print('now working on: '+j)
            filein=path+file+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
            f = open(filein, 'r')
            filedata = f.read()
            f.close()
            newdata2 = filedata.replace(usedtobe, shouldbe)
            #newdata = filedata.replace("0 1 0 1","0 0 1 1")#("green","blue")
            #newdata2=newdata.replace("green","blue")
            fileout= path+file+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
            f = open(fileout,'w')
            f.write(newdata2)
            f.close()'''

'''
#it works!!! follow it and then you will process it much faster!!!!!!!
filein= '/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
f = open(filein,'r')
filedata = f.read()
f.close()
'''

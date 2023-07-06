
#file = open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config')
'''
with open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('green', 'blue')
with open('file.txt', 'w') as file:
  file.write(filedata)
'''


import os, sys

path = '/home/jianning/blackbox/'#unit_box_0bluesmallsd311/'# "/var/www/html/"
dirs = os.listdir(path)#that meets my need!


for newshouldbe in ['t411','t412','t413','t416','t419','t421','t422','t423','t424','t431','t432','t433','t434']:
    oldreal = 'real'
    newusedtobe = newshouldbe + oldreal
    for file in dirs:
        #shouldbe = 't434real'
        if newshouldbe in file:
            print('the right file found!')
            print(file)
            for j in ['/model.config','/model.sdf']:
                print('now working on: '+j)
                filein=path+file+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
                f = open(filein, 'r')
                filedata = f.read()
                f.close()
                newdata=filedata.replace(newusedtobe+'r', newusedtobe)
                newdata2 = filedata.replace(newusedtobe, newshouldbe)
                #newdata = filedata.replace("0 1 0 1","0 0 1 1")#("green","blue")
                #newdata2=newdata.replace("green","blue")
                fileout= path+file+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
                f = open(fileout,'w')
                f.write(newdata2)
                f.close()


'''
for file in dirs:
    usedtobe="t434"
    shouldbe='t434real'
    if shouldbe in file:
        print(file)
        print('the right file found!')
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

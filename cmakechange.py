
#file = open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config')
'''
with open('/home/jianning/bluebox/unit_box_0bluesmallsd212/model.config', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('green', 'blue')
with open('file.txt', 'w') as file:
  file.write(filedata)
'''
import os, sys

path = '/home/jianning/gazebo_animatedbox_tutorial/'#'/home/jianning/model_editor_models/'#unit_box_0bluesmallsd311/'# "/var/www/html/"
dirs = os.listdir(path)#that meets my need!


for shouldbeold in ['CMakeLists']:#['n11','n12','n13','n14']:#,['t421','t422','t423','t431','t432','t433','t441','t442','443']:#['t411','t412','t413']:#,[,'t419','t421','t424','t431','t434']:
    for file in dirs:
        print(file)
        #print('the folder is loaded!')
        usedtobe1="float timefactor=10.0;"
        shouldbe1="float timefactor=5.0;"

        if shouldbeold in file:
            print('the file I need!')
            #print(file)
            #for j in ['/model.config','/model.sdf']:
            #print('now working on: '+j)
            filein=path+file#+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
            f = open(filein, 'r')
            filedata = f.read()
            f.close()

            newdata1 = filedata.replace('abn111', 'abn15')
            newdata2 = newdata1.replace('abn112', 'abn16')
            newdata3 = newdata2.replace('abn113', 'abn17')
            newdata4 = newdata3.replace('abn114', 'abn18')
            #newdata = filedata.replace("0 1 0 1","0 0 1 1")#("green","blue")
            #newdata2=newdata.replace("green","blue")
            '''
            file1=file[:-5]
            file2=file[-5:-3]
            file3=file[-3:]
            file2int=int(file2)
            file2int2=file2int+4#for 20.0#6#6 for 5.0#
            sfile2=str(file2int2)
            usedtobe2 = "ABn"+file2#"AnimatedBoxt41"+file2
            shouldbe2 = "ABn"+sfile2#"AnimatedBoxt41"+sfile2'''
            #newdata2 = newdata1.replace(usedtobe2, shouldbe2)
            fileout= path+file#1+sfile2+file3#+j#'/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
            f = open(fileout,'w')
            f.write(newdata4)
            f.close()

'''
#it works!!! follow it and then you will process it much faster!!!!!!!
filein= '/home/jianning/bluebox/unit_box_0bluesmallsd212/model.sdf'
f = open(filein,'r')
filedata = f.read()
f.close()
'''

'''
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
                f.close()'''

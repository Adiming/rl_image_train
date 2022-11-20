# rename image files
import os

path = os.getcwd()
path = os.path.join(path, "images")

fileList = os.listdir(path)

# for i in fileList:
#     name = i.split('.')
#     if len(name[0]) == 3:
#         num = int(name[0][2])
#         num-=1
#         newName = name[0][0] + name[0][1] + str(num) + 'c' + '.png'
        
#         os.rename(os.path.join(path,i), os.path.join(path,newName))

#     elif len(name[0]) ==4:
#         num = name[0][2]+name[0][3]
#         num = int(num)-1
#         newName = name[0][0] + name[0][1] + str(num) + 'c' + '.png'
#         os.rename(os.path.join(path,i), os.path.join(path,newName))
       
for i in fileList:
    name = i.split('.')
    if len(name[0]) == 4:
        newName = name[0][0] + name[0][1] + name[0][2] + '.png'
        
        os.rename(os.path.join(path,i), os.path.join(path,newName))

    elif len(name[0]) ==5:
        newName = name[0][0] + name[0][1] + name[0][2] + name[0][3] + '.png'
        os.rename(os.path.join(path,i), os.path.join(path,newName))
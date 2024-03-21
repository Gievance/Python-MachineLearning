# Writer：TuTTTTT
# 编写Time:2024/3/19 17:14
"""

torchvision.datasets中的COCO数据集需要安装COCO API

"""
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os

# 检查并创建COCO路径
abspath=f'D:/Program Files/DeepLearning/PytorchImple/' # 当前项目的绝对路径。【 必须修改！！！】
root=f'data/COCO/images'

# annfile=f'data/COCO/annotations'
annfile=f'data/COCO/annotations/stuff_val2017.json' #正确的指向了注释文件

root_path = os.path.join(abspath,root)
annfile_path= os.path.join(abspath,annfile)

"""
在创建路径时，应该使用 os.makedirs() 函数而不是 os.mkdir()，因为前者会递归创建多级目录，而后者只能创建单级目录。
报错1：PermissionError: [Errno 13] Permission denied: 'D:/Program Files/DeepLearning/PytorchImple/data/COCO/annotations'
    原因：annfile应该指向注释文件。
"""

if not os.path.exists(root_path): #
    os.makedirs(root_path)
if not os.path.exists(annfile_path):
    os.makedirs(annfile_path)

# 创建COCODetection对象->元组(图片,注释)
detect = dset.CocoDetection(root=root_path,annFile=annfile_path)

'''
写进文件也没啥用，看不明白，那就debug
'''
# with open('./file_info.txt','w+') as file:
#     image,info=detect[0]
#     file.write(str(image))
#     file.write('\n')
#     file.write(str(info))

# debug后知晓annfile的json结构树
# print(detect[0])


'''
接下来将bbox显示到图片上
'''

from PIL import ImageDraw
image,info=detect[2]
image_bbox=ImageDraw.ImageDraw(image)

# info包括多个注释,逻辑是遍历每个注释中的bbox字段
for anno in info:
    x_min,y_min,width,height=anno['bbox']
    image_bbox.rectangle(((x_min,y_min),(x_min+width,y_min+height)),outline='red',width=2)

image.save('output.jpg')

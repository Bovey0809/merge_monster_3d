# %%
import numpy as np

li = './work_dirs/nanodet/lian.npy'
my = './my.npy'
liimg = './work_dirs/nanodet/lian_img.npy'
myimg = './my_img.npy'

li = np.load(li)
my = np.load(my)

print(np.allclose(li, my, 1e-1))

liimg = np.load(liimg)
myimg = np.load(myimg)
# %%
print(np.allclose(liimg, myimg, 1e-2))
li32 = './work_dirs/nanodet/lian32.npy'
my32 = './myfeat32.npy'

li32 = np.load(li32)
my32 = np.load(my32)
print(np.allclose(li32, my32, 1e-2))
# %%
    
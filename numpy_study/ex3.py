import numpy as np


# 3. 如何取值?

def pprint(*str):
    print("#" * 10)
    print(*str)


a = np.arange(0, 10)
pprint(a)
pprint(a[0])
a2 = a.reshape([2, 5])
pprint(a2)
# pprint(a2[0:2][1:4]) # 取法不对

pprint(a2[0, 1])  # axis=0里面的0, axis=1里面的1

pprint(a2[1, 0])  # axis=0里面的1, axis=1里面的0

pprint(a2[:, 0])  # list[:] axis=0里面的所有，axis=1里面所有的0

pprint(a2[1:, -1])

pprint(a2[:, ::2])  # list[::2]每2取1个
# ndarray[axis0start:axis0end:step0, axis1start:axis1end:step0]

pprint("*" * 50)

num2 = np.random.randn(4, 640, 640, 3)  # 4张640x640x3的图片
# 取第1张图片
pprint(num2[0].shape)
pprint(num2[0, ...].shape)

# 取第1张和最后1张图片
pprint(num2[[0, -1], ...].shape)  # 2 x 640 x 640 x 3

# 第2张和第3张图片
pprint(num2[1:3, ...].shape)

# 每2张图片来取
pprint(num2[::2, ...].shape)

# 4x320x320x3
pprint(num2[:, :320, :, :].shape)

pprint(num2[:, :320, :320, :].shape)

pprint(num2[:, ::2, ::2, :].shape)

pprint(num2[0, 0, ...].shape)  # 640x3

pprint(num2[0, :, 0, ...].shape)  # 640x1x3->640x3
# 4 x 640x640x3
pprint(num2[..., 0].shape)  # 4x 640 x 640

pprint(num2[:2, :640:2, :320:2, 1:3].shape)  # 2 x 320 x 160 x 2

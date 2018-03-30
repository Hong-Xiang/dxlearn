import imageio
import numpy as np
num_file = 1
for i in range(num_file):
    fileholder = 'debug'
    img = np.load('./{}/effmap_{}.npy'.format(fileholder, i))
    print('img max:', np.max(img))
    print(img.shape)
    # img = img.reshape([320, 320, 160])
    imgslice = img[:,80,:]
    print(np.max(imgslice))
    print(np.min(imgslice))
    # imgslice = imgslice - np.min(imgslice)
    # imgslice = imgslice / np.max(imgslice)
    imageio.imwrite('./{}/recon_side_{}.png'.format(fileholder, i), imgslice)
    imgslice = img[79,:,:]
    print(np.max(imgslice))
    print(np.min(imgslice))
    # imgslice = imgslice - np.min(imgslice)
    # imgslice = imgslice / np.max(imgslice)
    imageio.imwrite('./{}/recon_{}.png'.format(fileholder, i), imgslice)
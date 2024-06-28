# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:59:51 2024

@author: nlinares
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy.io import fits
import time
import cv2

filename_slopes = 'ol_residual_slopes_7dir_corrClip_0_20240514.fits'
filename_pmx = 'pmx_7dir_100_modes_20240424_104232.fits'
file_sun = 'sun_granullation.fits'
file_zernikes = 'mode2acts_zern_100.fits'

# Parameters
nDirs = 7
img_size = 512
acts_config = [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 30, 30,
               28, 28, 26, 24, 22, 20, 18, 14, 10]
image_sampling = 50 # 1 every N samples will have an image computed
tt_margin = 50

gain = 0.2
decay = 0.9999
delay = 2 # number of samples
# Generate binary pupil

binaryPupil = np.zeros((np.max(acts_config), np.max(acts_config)))

for i in range(len(binaryPupil)):
    beginInt = (len(binaryPupil) - acts_config[i]) / 2
    endInt = (len(binaryPupil) - acts_config[i]) / 2 + acts_config[i]
    for j in range(len(binaryPupil)):
        if (j >= beginInt) and (j < endInt):
            binaryPupil[i, j] = True

# Load data
slopes_OL = fits.open(filename_slopes)[0].data
# slopes_OL = slopes_OL[0:5400, :]
pmx_glao = fits.open(filename_pmx)[0].data
sun_image = fits.open(file_sun)[0].data
zernikes = fits.open(file_zernikes)[0].data

sun_image = sun_image[round(sun_image.shape[0]//2 - img_size/2):round(sun_image.shape[0]//2 + img_size/2),
            round(sun_image.shape[1]//2 - img_size/2):round(sun_image.shape[1]//2 + img_size/2)]

sun_image = np.pad(sun_image, img_size // 2)

# Convert Zernikes to 2D

zernikes = zernikes[:, ~np.all(zernikes == 0, axis=0)]
zernikes_2D = np.zeros((zernikes.shape[1], np.max(acts_config), np.max(acts_config)))

k = 0

for i in range(np.max(acts_config)):
    for j in range(np.max(acts_config)):
        if binaryPupil[i, j]:
            zernikes_2D[:, i, j] = zernikes[k, :]

            k = k + 1

zernikes_2D = zernikes_2D * ((20.0 * 2 * np.pi) / 0.525) # From |Z| <= 1 to radians

gray_image = sun_image[round(sun_image.shape[0]//2 - img_size/2 + tt_margin):round(sun_image.shape[0]//2 + img_size/2-tt_margin),
                       round(sun_image.shape[1]//2 - img_size/2 + tt_margin):round(sun_image.shape[1]//2 + img_size/2-tt_margin)].astype(float)
# laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
#
# Cini = laplacian.var()
Cini = np.std(gray_image) / np.mean(gray_image)
print(Cini)


# Split data
nCents = slopes_OL.shape[1] // (nDirs*2)
nSamples = slopes_OL.shape[0]
nModes = pmx_glao.shape[0]

slopes_x = np.zeros((nDirs, nSamples, nCents))
slopes_y = np.zeros((nDirs, nSamples, nCents))

for i in range(nDirs):
    slopes_x[i, :, :] = slopes_OL[:, i:-1:nDirs*2]
    slopes_y[i, :, :] = slopes_OL[:, i+1:-1:nDirs*2]

# Compute modes GLAO

rmx_glao = np.linalg.pinv(pmx_glao, 0.0025).T
rmx_glao = rmx_glao[~np.all(rmx_glao == 0, axis=1)]

modes_glao = np.zeros((nSamples, rmx_glao.shape[0]))

for j in range(nSamples):
    modes_glao[j,:] = np.matmul(rmx_glao, np.squeeze(slopes_OL[j, :]))

# Compute modes per dir and sample

t = time.time()
rmx_dir = np.zeros((nDirs, rmx_glao.shape[0], slopes_OL.shape[1] // nDirs))
pmx_dir = np.zeros((nDirs, rmx_glao.shape[0], slopes_OL.shape[1] // nDirs))

modes_dir = np.zeros((nDirs, nSamples, rmx_glao.shape[0]))
slopes_OL_perdir = np.zeros((nDirs, nSamples, slopes_OL.shape[1] // nDirs))

# print(rmx_dir.shape, pmx_dir.shape, modes_dir.shape, slopes_OL_perdir.shape)

mask_base = np.zeros((nDirs*2,))
mask_base[0] = mask_base[1] = 1
mask_vector = np.tile(mask_base,(1, nCents)).astype(bool)  # Same dims as 1 vector
mask_matrix = np.tile(mask_base,(nModes,nCents)).astype(bool)  # Same dims as pmx_glao

for i in range(nDirs):
    pmx_dir_temp = pmx_glao * np.roll(mask_matrix, i*2)
    pmx_dir_temp = pmx_dir_temp[:,~np.all(pmx_dir_temp == 0, axis=0)]  # Remove 0 cols
    pmx_dir[i, :, :] = pmx_dir_temp[~np.all(pmx_dir_temp == 0, axis=1)]  # Remove 0 rows

    rmx_dir[i, :, :] = np.linalg.pinv(pmx_dir[i, :, :], 0.0025).T

    for j in range(nSamples):
        slopes_OL_crop = slopes_OL[j, :]
        slopes_OL_crop = slopes_OL_crop * np.roll(mask_vector, i*2)
        slopes_OL_perdir[i, j, :] = np.squeeze(slopes_OL_crop[:, ~np.all(slopes_OL_crop == 0, axis=0)])
        modes_dir[i, j, :] = np.matmul(rmx_dir[i, :, :], slopes_OL_perdir[i, j, :])

print('Time computing modes per dir: ' + str(time.time() - t) + '[s]')
# Simulate OL Science

t = time.time()

sun_image_aberrated_perdir_OL = np.zeros((nDirs, nSamples // image_sampling, img_size, img_size))
contrast_OL_perdir = np.zeros((nDirs, nSamples // image_sampling))

rini = round(sun_image.shape[0]//2 - img_size/2)
rend = round(sun_image.shape[0]//2 + img_size/2)

cini = round(sun_image.shape[1]//2 - img_size/2)
cend = round(sun_image.shape[1]//2 + img_size/2)

index = 0
for s in range(0, nSamples, image_sampling):
    # print('Samples: ' + str(s))

    for dir in range(nDirs):
        phi = np.einsum('ijk, i->jk', zernikes_2D, modes_dir[dir, s, :])

        pupil = binaryPupil * np.exp((0.+1j) * phi)

        ft = np.fft.fft2(pupil) / pupil.size
        psf = np.pad(np.fft.fftshift(np.real(np.conj(ft) * ft)), (2*img_size - ft.shape[0])//2 )

        sun_image_aberrated_t = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(sun_image) * np.fft.fft2(psf))))

        sun_image_aberrated_perdir_OL[dir, index, :, :] = sun_image_aberrated_t[rini:rend, cini:cend]

        laplacian = cv2.Laplacian(sun_image_aberrated_perdir_OL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin].astype(float), cv2.CV_64F)

        contrast_OL_perdir[dir, index] = np.std(sun_image_aberrated_perdir_OL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin]) / \
                                         np.mean(sun_image_aberrated_perdir_OL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin])
        # contrast_OL_perdir[dir, index] = laplacian.var()
    index += 1

print('Time computing images OL, all directions: ' + str(time.time() - t) + '[s]')
plt.plot(contrast_OL_perdir.T)
labels = [f'Dir {i}' for i in range(contrast_OL_perdir.shape[0])]
plt.legend(labels)
plt.ylabel('Contrast')
plt.xlabel('Sample')
plt.grid()
plt.show()

###### Simulate CL Science

t = time.time()

sun_image_aberrated_perdir_CL = np.zeros((nDirs, nSamples // image_sampling, img_size, img_size))
modes_res_CL_perdir_glao = np.zeros_like(modes_dir)
contrast_CL_perdir = np.zeros((nDirs, nSamples // image_sampling))

# Compute CL

uprev = np.zeros((delay+1, rmx_glao.shape[0]))  # Last row is the newest

for s in range(nSamples):
    if s < delay:  # before the delay sample, the system does not respond to any variation
        u = uprev[-1, :]
    else:
        # Compute error
        error = modes_glao[s-delay, :] - uprev[-1 * (delay + 1)]
        # Compute control action
        u = uprev[-1, :] * decay + gain * error  # modal correction, negative feedback is explicitly included
    # Compute residual per dir
    for dir in range(nDirs):
        modes_res_CL_perdir_glao[dir, s, :] = modes_dir[dir, s, :] - u
    # Update variables for next iteration
    for i in range(delay):
        uprev[i, :] = uprev[i+1, :]

    uprev[-1, :] = u

index = 0

for s in range(0, nSamples, image_sampling):
    # print('Samples: ' + str(s))

    for dir in range(nDirs):
        phi = np.einsum('ijk, i->jk', zernikes_2D, modes_res_CL_perdir_glao[dir, s, :])

        pupil = binaryPupil * np.exp((0.+1j) * phi)

        ft = np.fft.fft2(pupil) / pupil.size
        psf = np.pad(np.fft.fftshift(np.real(np.conj(ft) * ft)), (2*img_size - ft.shape[0])//2 )

        sun_image_aberrated_t = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(sun_image) * np.fft.fft2(psf))))

        sun_image_aberrated_perdir_CL[dir, index, :, :] = sun_image_aberrated_t[rini:rend, cini:cend]

        laplacian = cv2.Laplacian(sun_image_aberrated_perdir_CL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin].astype(float), cv2.CV_64F)

        contrast_CL_perdir[dir, index] = np.std(sun_image_aberrated_perdir_CL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin]) / \
                                         np.mean(sun_image_aberrated_perdir_CL[dir, index, tt_margin:-tt_margin, tt_margin:-tt_margin])

        # contrast_CL_perdir[dir, index] = laplacian.var()

    index += 1

print('Time computing images CL, all directions: ' + str(time.time() - t) + '[s]')
plt.plot(contrast_CL_perdir.T)
labels = [f'Dir {i}' for i in range(contrast_CL_perdir.shape[0])]
plt.legend(labels)
plt.ylabel('Contrast')
plt.xlabel('Sample')
plt.grid()
plt.show()
print(Cini)

# Reproduce data

fig, axs = plt.subplots(1, 2, figsize=(8, 8))

lim = np.max(np.abs(modes_dir[:]))
lim_cl = np.max(np.abs(modes_res_CL_perdir_glao[:]))

fig.tight_layout()

def animate(s):
    for i in range(nDirs):
        axs[0].clear()
        axs[1].clear()

    axs[0].imshow(sun_image_aberrated_perdir_OL[6, s, :, :], 'gray')

    axs[1].imshow(sun_image_aberrated_perdir_CL[6, s, :, :], 'gray')
    fig.suptitle(f'Sample {s*image_sampling}')

animation = FuncAnimation(fig, animate, frames=range(0, nSamples // image_sampling, 1))
plt.show()
print(np.mean(contrast_CL_perdir[:,2:], axis=1), np.mean(contrast_OL_perdir[:,2:], axis=1))

fits.writeto('ol_sun.fits', sun_image_aberrated_perdir_OL)
fits.writeto('cs_results_leaky.fits', sun_image_aberrated_perdir_CL)


print('End')
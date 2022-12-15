#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import cv2 as cv
import numpy as np

import image as img

# the directory this script is running form
WORKING_DIR: str = os.path.normpath(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        './',
    )
)

################################################################################
#                this file is just an illustration, it is meant                #
#              to be READ, then MODIFIED to suit your application              #
################################################################################


# directory of input images
input_photo_dir: str = 'input_tiff/size_full'


def main():
    # gather input filenames
    input_file_list = os.listdir(os.path.join(WORKING_DIR, input_photo_dir))
    for i in range(len(input_file_list)):
        # prefix filenames with full path
        input_file_list[i] = os.path.join(WORKING_DIR, input_photo_dir, input_file_list[i])
    # filter out previously saved align matrix files if exist
    input_file_list = list(
        e for e in input_file_list if not e.endswith(img.Stack.ALIGN_MATRIX_FILE_EXTENSION)
    )
    # create stack object from list of files
    stack = img.Stack(input_file_list)
    # align them, this may take hours if there is a lot of images
    stack.align()
    #stack.align(filter_=False)

    # After potentially hours of alignment, it is recommended to save the
    # alignment result, since the following steps need a lot of memory and the
    # program might crash due to lack of memory.
    # Basically the amount of memory you need is around the total size of
    # input image files.  Of course you can split the job into multiple parts,
    # or allocate a ridiculously large swap file to get around this.
    stack.write_align_matrix_to_files()

    # Take the statistic mean over all aligned images, this gives a 'noise free' image.
    aligned_mean = stack.statistics(img.Stack.TYPE.MEAN, return_same_dtype=False)
    #aligned_mean = stack.budget_statistics(img.Stack.TYPE.MEAN)
    #np.save(os.path.join(WORKING_DIR, '_aligned_mean.npy'), aligned_mean, allow_pickle=False)
    #
    # Take the statistic median over all UNALIGNED images.  If the time span of
    # all images are long enough, this should give the structureless background
    # (e.g. light pollution, vignetting etc.).
    unaligned_median = stack.statistics(img.Stack.TYPE.MEDIAN, aligned=False, return_same_dtype=False)
    #unaligned_median = stack.budget_statistics(img.Stack.TYPE.MEDIAN_OF_MEDIANS, aligned=False)
    #np.save(os.path.join(WORKING_DIR, '_unaligned_median.npy'), unaligned_median, allow_pickle=False)
    #
    # Blur this 'background' to smooth it out, otherwise any brightness fluctuation
    # will have a great impact on the subtracted image if its exposure value is to
    # be greatly boosted.
    unaligned_median_blur = cv.blur(unaligned_median, (61, 61))

    # NOTES on time saving:
    # To achieve a good structureless background, you need a 'long time span' sequence of images.
    # Since aligning images are very time consuming, it will be way more practical to do two
    # separate runs.  One with few but sufficient images for aligned mean, another use all the
    # images for unaligned median.
    # (read saved images from earlier run with command
    # `mean = cv.imread(mean_img_path, cv.IMREAD_UNCHANGED)` )
    #
    # To avoid memory shortage for 'background' calculation with all the images, you can run it on
    # down scaled images.  For example, export your images from Lightroom with 1/8 of the original
    # size, obtain `unaligned_median` with it, then up scale it with the following command:
    # `unali_medi = cv.resize(unali_medi, dsize=(0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)`
    # (do this **BEFORE** bluring).

    # Subtract the 'background' from the image, bring out originally washed out
    # structures (like Milky Way).  The exposure value of these images are meant
    # to be stretched, use `Image.stretch()` function or any program you like
    # (e.g. darktable, Lightroom).
    #
    # Play with the value, and pick the one that works.
    # (more than 90 might be too extream)
    subtracted_70 = aligned_mean - 0.7 * unaligned_median_blur
    subtracted_80 = aligned_mean - 0.8 * unaligned_median_blur
    subtracted_90 = aligned_mean - 0.9 * unaligned_median_blur
    # # Since the data type is unsigned int, when the subtractor is greater than
    # # the 'subtractee', the value will flip to maximum.  This part is just to
    # # take care of that.
    # subtracted_70[aligned_mean < subtracted_70] = 0
    # subtracted_80[aligned_mean < subtracted_80] = 0
    # subtracted_90[aligned_mean < subtracted_90] = 0

    # Write produced images to files.
    cv.imwrite(os.path.join(WORKING_DIR, '_aligned_mean.tiff'), img.Image.clip(aligned_mean))
    cv.imwrite(os.path.join(WORKING_DIR, '_unaligned_median.tiff'), img.Image.clip(unaligned_median))
    cv.imwrite(os.path.join(WORKING_DIR, '_unaligned_median_blur.tiff'), img.Image.clip(unaligned_median_blur))
    cv.imwrite(os.path.join(WORKING_DIR, 'subtracted_70.tiff'), img.Image.stretch(subtracted_70, extra_factor=2))
    cv.imwrite(os.path.join(WORKING_DIR, 'subtracted_80.tiff'), img.Image.stretch(subtracted_80, extra_factor=2**2))
    cv.imwrite(os.path.join(WORKING_DIR, 'subtracted_90.tiff'), img.Image.stretch(subtracted_90, extra_factor=2**4))

    # NOTES on color management:
    # opencv dose not care about color space, it just reads and writes pixel value, and the file
    # written with `cv.imwrite()` does not have any color profile attached.  To preserve color
    # profile, you can extract it to a separate file using other tools, then embed it back later.
    #
    # I recommend the [`exiftool`](https://exiftool.org/) command-line tool:
    # extract: `exiftool -icc_profile -b -w icc ${image_file}`
    # embed  : `exiftool '-ICC_Profile<=${icc_file}' ${image_file}`


if __name__ == '__main__':
    # main()
    print('READ this file first!  It is actually really short.')
    input('Script finished, press <Enter> to exit.')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import cv2 as cv

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
    # create stack object from list of files
    stack = img.Stack(input_file_list)
    # align them, this may take hours if there is a lot of images
    stack.align()

    # # After hours of alignment, it is recommended to save the aligned result,
    # # since the following steps are very memory heavy and the program may quit
    # # due to lack of system memory.
    # # Basically the memory you need is around the same size of the total input
    # # image files.  Of course you can split the job into multiple parts to get
    # # around this.
    # #
    # # But write back to original file is a destructive process, that is why
    # # this command is commented out.
    # stack.write_aligned_back_to_files()

    # Take the statistic mean over all aligned images, this gives a 'noise free' image.
    aligned_mean = stack.statistics(img.Stack.TYPE.MEAN)
    # Take the statistic median over all UNALIGNED images.  If the time span of
    # all images are long enough, this should give the structureless background
    # (e.g. light pollution, vignetting etc.).
    unaligned_median = stack.statistics(img.Stack.TYPE.MEDIAN, aligned=False)
    # Blur this 'background' to smooth it out, otherwise any brightness fluctuation
    # will have a great impact on the subtracted image if its exposure value is to
    # be greatly boosted.
    unaligned_median_61 = cv.blur(unaligned_median, (61, 61))

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
    # to be boosted, use any program you like (e.g. Lightroom).
    #
    # Play with the value, and pick the one that works.
    # (lower than 10 might be too extream)
    subtracted_00 = aligned_mean - unaligned_median
    subtracted_05 = (aligned_mean - (1 - 0.05) * unaligned_median_61).astype(aligned_mean.dtype)
    subtracted_10 = (aligned_mean - (1 - 0.10) * unaligned_median_61).astype(aligned_mean.dtype)
    subtracted_15 = (aligned_mean - (1 - 0.15) * unaligned_median_61).astype(aligned_mean.dtype)
    subtracted_20 = (aligned_mean - (1 - 0.20) * unaligned_median_61).astype(aligned_mean.dtype)
    # Since the data type is unsigned int, when the subtractor is greater than
    # the 'subtractee', the value will flip to maximum.  This part is just to
    # take care of that.
    subtracted_00[aligned_mean < subtracted_00] = 0
    subtracted_05[aligned_mean < subtracted_05] = 0
    subtracted_10[aligned_mean < subtracted_10] = 0
    subtracted_15[aligned_mean < subtracted_15] = 0
    subtracted_20[aligned_mean < subtracted_20] = 0

    # Write produced images to files.
    cv.imwrite(os.path.join(WORKING_DIR, 'output_aligned_mean.tiff'), aligned_mean)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_unaligned_median.tiff'), unaligned_median_61)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_subtracted_00.tiff'), subtracted_00)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_subtracted_05.tiff'), subtracted_05)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_subtracted_10.tiff'), subtracted_10)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_subtracted_15.tiff'), subtracted_15)
    cv.imwrite(os.path.join(WORKING_DIR, 'output_subtracted_20.tiff'), subtracted_20)

    # NOTES on color management:
    # opencv dose not care about color space, it just read and write pixel values,and the file
    # written by `cv.imwrite()` does not have any color profile attached.  To preserve color
    # profile, you can extract it into a separate file using other tools, then embed it back.
    #
    # I recommend the [`exiftool`](https://exiftool.org/) command-line tool:
    # extract: `exiftool -icc_profile -b -w icc ${image_file}`
    # embed  : `exiftool '-ICC_Profile<=${icc_file}' ${image_file}`


if __name__ == '__main__':
    # main()
    print('READ this file first!  It is actually really short.')
    input('Script finished, press <Enter> to exit.')

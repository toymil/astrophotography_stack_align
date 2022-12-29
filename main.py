#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os

import cv2 as cv
import numpy as np

import image as img


################################################################################
#                this file is just an illustration, it is meant                #
#              to be READ, then MODIFIED to suit your application              #
################################################################################

# directories
working_dir = os.getcwd()
input_dir: str = r""
output_dir: str = r""
def p(path: str, working_dir: str = working_dir) -> str:
    return os.path.normpath(os.path.join(working_dir, path))


def main():
    # gather input filenames
    input_file_list = os.listdir(p(input_dir))
    for i in range(len(input_file_list)):
        # prefix filenames with full path
        input_file_list[i] = os.path.join(p(input_dir), input_file_list[i])

    # create stack object from list of files
    stack = img.Stack(input_file_list)

    # load previously (auto) saved align progress
    #stack.load_align_matrix_from_file()

    # align them, this may take hours if there is a lot of images
    stack.align()
    #stack.align(filter_=False)

    # Take the statistic median over all UNALIGNED images.  If the time span of
    # all images are long enough, this should give the structureless background
    # e.g. light pollution. (vignette should be removed BEFOREHAND)
    unaligned_median = stack.statistics(img.Stack.TYPE.MEDIAN_OF_MEDIANS, aligned=False, memory_budget=4)
    np.save(os.path.join(p(output_dir), '_unaligned_median.npy'), unaligned_median, allow_pickle=False)

    # # Blur this 'background' to smooth it out, otherwise any brightness
    # # fluctuation will have a great impact on the subtracted image if its
    # # exposure value is to be greatly boosted.
    # unaligned_median_blur = cv.blur(unaligned_median, (61, 61))

    # NOTES on time saving:
    # To achieve a good structureless background, you need a 'long time span' sequence of images.
    # Since aligning images are very time consuming, it will be way more practical to do two
    # separate runs.  One with few but sufficient images for aligned mean, another use all the
    # images for unaligned median.
    # (read saved images from earlier run with command
    # `mean = cv.imread(mean_img_path, cv.IMREAD_UNCHANGED)` )
    #
    # To avoid memory shortage for 'background' calculation with all the images, you can run it on
    # down scaled images.  For example, export your images with 1/8 of the original size, obtain
    # `unaligned_median` with it, then up scale it with the following command:
    # `unali_medi = cv.resize(unali_medi, dsize=(0, 0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)`
    # (do this **BEFORE** bluring).

    # Take the statistic mean over all aligned images, this gives a 'noise free' image.
    aligned_mean_090 = stack.statistics(
        img.Stack.TYPE.MEAN,
        # subtract the 'background' from each source frame, play with the
        # factor, and pick one that works.
        preprocess=lambda x: img.IIO.subtract_image(x, unaligned_median, 0.9),
        memory_budget=4,
    )
    np.save(os.path.join(p(output_dir), '_aligned_mean_090.npy'), aligned_mean_090, allow_pickle=False)

    # Write produced image to files.
    cv.imwrite(
        os.path.join(p(output_dir), '_unaligned_median.tiff'),
        img.Image.clip(
            unaligned_median,
            np.uint16,
        ),
    )
    cv.imwrite(
        os.path.join(p(output_dir), '_aligned_mean_090.tiff'),
        # Stretch exposure.  If you want to do this in another program (e.g.
        # darktable, RawTherapee, Lightroom etc.), switch to the `clip`
        # function.
        img.Image.stretch(
            aligned_mean_090,
            np.uint16,
            extra_factor=2**4,
        ),
    )

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

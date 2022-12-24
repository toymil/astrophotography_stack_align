#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import concurrent.futures
import enum
import math
import os
import random
import warnings
from collections.abc import Callable, Sequence
from typing import TypeVar

import cv2 as cv
import numpy as np
import pywt

T = TypeVar('T')


class Image:
    ALIGN_TRANS_MATRIX_FILE_EXT = '.align.npy'

    def __init__(
        self,
        image_file_path: str,
    ) -> None:
        self.path: str = image_file_path

        self.image: np.ndarray = None
        self._image_gray_float_wlred: np.ndarray = None
        self._image_gray_blur_float_wlred: np.ndarray = None

        self.stars: tuple[tuple[
            np.ndarray,  # S2A[x, y]
            float,  # brightness
        ], ...] = None

        self.star_structure_neighbour_range_low: float = None
        self.star_structure_neighbour_range_high: float = None
        self.structures: tuple[tuple[
            np.ndarray,  # S2A[x, y]; structure source star centroid
            np.ndarray,  # SNx2A[ [angle, ratio] ]; feature array of triangles
        ], ...] = None

        # transformation matrix to align with another image
        self.align_trans_matrix: np.ndarray = None  # S3x3A

    def load(self, *, compute_wlred: bool = False) -> None:
        self.image = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        if compute_wlred:
            image_gray: np.ndarray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            # use float64 since the number is small, might encounter the
            # 'rounding to 0' problem (if used float32, this is exactly
            # what will happen in the `cv.moments()` part later)
            if image_gray.dtype == np.float64:
                image_gray_float = image_gray
            elif image_gray.dtype in (np.float16, np.float32):
                image_gray_float = image_gray.astype(np.float64, casting='safe')
            else:
                image_gray_float: np.ndarray = (
                    image_gray.astype(np.float64, casting='safe') / np.iinfo(image_gray.dtype).max
                )
            self._image_gray_float_wlred = Image.wavelet_dec_red_rec(image_gray_float)

            # TODO: tune radius and sigma?
            image_gray_blur: np.ndarray = cv.GaussianBlur(image_gray, (9, 9), 0, 0)
            if image_gray_blur.dtype == np.float64:
                image_gray_blur_float = image_gray_blur
            elif image_gray_blur.dtype in (np.float16, np.float32):
                image_gray_blur_float = image_gray_blur.astype(np.float64, casting='safe')
            else:
                image_gray_blur_float: np.ndarray = (
                    image_gray_blur.astype(np.float64, casting='safe') / np.iinfo(image_gray_blur.dtype).max
                )
            self._image_gray_blur_float_wlred = Image.wavelet_dec_red_rec(image_gray_blur_float)

    def release(self) -> None:
        self.image = None
        self._image_gray_float_wlred = None
        self._image_gray_blur_float_wlred = None

    def set_star_structure_neighbour_range(self, low: float, high: float) -> None:
        self.star_structure_neighbour_range_low = low
        self.star_structure_neighbour_range_high = high

    def set_trans_matrix(self, trans_matrix: np.ndarray) -> None:
        self.align_trans_matrix = trans_matrix

    def save_trans_matrix(self) -> bool:
        if self.align_trans_matrix is None:
            return False
        else:
            np.save(
                os.path.splitext(self.path)[0] + Image.ALIGN_TRANS_MATRIX_FILE_EXT,
                self.align_trans_matrix,
                allow_pickle=False,
            )
            return True

    def load_trans_matrix(self) -> bool:
        path: str = os.path.splitext(self.path)[0] + Image.ALIGN_TRANS_MATRIX_FILE_EXT
        if os.path.isfile(path):
            self.set_trans_matrix(np.load(
                path,
                allow_pickle=False,
            ))
            return True
        else:
            return False

    def compute(self, filter_: bool) -> None:
        self.detect_stars(filter_by_brightness=filter_)
        if not filter_:
            self.set_star_structure_neighbour_range(0, np.linalg.norm(self.image.shape[:2]))
        self.form_structures()

    @staticmethod
    def wavelet_dec_red_rec(
        image: np.ndarray,
        decomposition_level: int = 5,
        remove_to_small_scale_layer: int = -1,
        remove_large_scale: bool = True,
    ) -> np.ndarray:
        # decomposition
        # TODO: what is 'db8' and why?
        coeffs = pywt.wavedec2(image, 'db8', level=decomposition_level)
        # reduction
        for n in range(remove_to_small_scale_layer, 0):
            for i in range(0, 3):
                coeffs[n][i].fill(0)
        if remove_large_scale:
            coeffs[0].fill(0)
        # reconstruction
        return pywt.waverec2(coeffs, 'db8')

    def detect_stars(self, filter_by_brightness: bool) -> None:
        # threshold to binary for contour detection
        image_gray_blur_float_wlred_binary: np.ndarray = cv.threshold(
            (t := self._image_gray_blur_float_wlred),
            # Dim white blob on black background, `mean()` is no different from
            # black (black is not necessarily 0), and `std()` is super small.
            # TODO: How to threshold relatively outstandingly bright stars?
            # Assume stars are just below overexposed in the original image,
            # then its value will be at most 1 in the float image, related to
            # the gaussian blur parameters and star size in the blur float
            # image.
            # We can estimate a threshold value from the general star size (of
            # the not blured image) and gaussian blur parameters.  This
            # threshold value should grow with the general star size, but will
            # have an upper limit, since given a certain gaussian blur
            # parameters the blured value of star 'edges' does not grow over a
            # certain star size.
            #
            # I will just take a guess for now :(
            t.mean() + 0.135,
            255,
            cv.THRESH_BINARY,
        )[1].astype(np.uint8)

        # find contours
        contours = cv.findContours(
            image_gray_blur_float_wlred_binary,
            cv.RETR_LIST,
            cv.CHAIN_APPROX_SIMPLE,
        )[0]

        # characterize contours to stars
        stars: list[tuple[
            np.ndarray,  # S2A[x, y]
            float,  # brightness
        ]] = []
        for c in contours:
            cb: tuple[  # contour box
                int,  # x
                int,  # y
                int,  # width (x)
                int,  # height (y)
            ] = cv.boundingRect(c)
            cicb = c - cb[:2]  # contour in contour box
            # calculate centroid from clear image
            star_mask: np.ndarray = cv.drawContours(
                # image mask for brightness weighted moments calculation
                # draw filled (value 1) contours on black (value 0) image
                # note that ndarray dimension takes [y, x]
                np.zeros(cb[:-3:-1], self._image_gray_float_wlred.dtype),
                [cicb], 0,
                1, cv.FILLED,
            )
            M: dict[str, float] = cv.moments(
                # `dtype` precision used in moments calculation should
                # be at least `np.float64`, `np.float32` will likely to
                # encounter 'rounded to 0' problem
                star_mask
                * self._image_gray_float_wlred[
                    cb[1] : cb[1] + cb[3],
                    cb[0] : cb[0] + cb[2]
                ]
            )
            centroid = cb[:2] + np.array([ M['m10'] / M['m00'], M['m01'] / M['m00'] ])
            # denote brightness from blur image, increase robustness
            brightness = self._image_gray_blur_float_wlred[int(centroid[1]), int(centroid[0])]

            stars.append( (centroid, brightness) )

        # sort and filter by brightness
        stars.sort(key=lambda e: e[1], reverse=True)
        # All stars in this list is already 'well defined' if the thresholding
        # process is robust.  But to save computing time in structures forming
        # and inter image structure matching, we only take the brightest among
        # them.
        if filter_by_brightness:
            brightness_cutoff = np.array(tuple(e[1] for e in stars)).mean()
            # This cutoff value is somewhat arbitrary, since the brightness of stars
            # does not form a good distribution.  Anyway this is just a temporary
            # solution, see the 'to do' below.
            for i in range(len(stars)):
                if stars[i][1] < brightness_cutoff:
                    stars = stars[:i]
                    break
        # TODO: sort, but do not filter stars.  To save computing time, form
        # structures in large scale for the brightest stars like usual (lot of
        # triangle features), but only form structures with the nearest N
        # neighbours for 'dim' stars (small number of triangle features).

        self.stars = tuple(stars)

    @staticmethod
    def iof(l: Sequence[T], i: int) -> T:
        # index overflow / circular linked list
        while True:
            if -1 < i < len(l):
                break
            else:
                i -= (i // len(l)) * len(l)
        return l[i]

    def form_structures(
        self,
        # these 2 coeffs will be used to calculate range_low/high if they are not set
        star_structure_neighbour_range_low_coeff: float = 0.1,
        star_structure_neighbour_range_high_coeff: float = 1.0,
    ) -> None:
        # TODO: move the description in jupyter notebook here around corresponding codes

        if (
            (self.star_structure_neighbour_range_low is None)
            or (self.star_structure_neighbour_range_high is None)
        ):
            warnings.warn('Star structure neighbour range low/high not set, calculate with coeff.')
            # images to align should have the same `star_structure_neighbour_range_low/high`
            # as the reference image
            (
                self.star_structure_neighbour_range_low,
                self.star_structure_neighbour_range_high
            ) = np.array(
                # use coeff to multiply the std of star separation
                [
                    star_structure_neighbour_range_low_coeff,
                    star_structure_neighbour_range_high_coeff
                ]
            ) * np.array(
                # array of distances from the brightest star to all other stars
                tuple( np.linalg.norm(s[0] - self.stars[0][0]) for s in self.stars[1:] )
            ).std()

        structures: list[tuple[
            np.ndarray,  # S2A[x, y]; structure source star centroid
            np.ndarray,  # SNx2A[ [angle, ratio] ]; feature array of triangles
        ]] = []

        for s1 in self.stars:
            neighbours: list[
                tuple[  # this will actually be list because it will be constructed part by part
                    np.ndarray,  # S2A(separation vector)
                    float,  # magnitude of separation vector
                    float,  # angle from angle reference vector
                ]
            ] = []

            for s2 in self.stars:
                if (
                    self.star_structure_neighbour_range_low
                    < np.linalg.norm( (sv := (s2[0] - s1[0])) )
                    < self.star_structure_neighbour_range_high
                ):
                    # populate separation vector, magnitude
                    neighbours.append([sv, (sr := np.linalg.norm(sv))])
                    # `neighbours` is sorted because `stars` is sorted.  So
                    # `neighbour[0][0]` is the angle reference vector.
                    # populate angle
                    neighbours[-1].append(
                        np.arccos(
                            np.clip(
                                np.dot(neighbours[0][0], sv) / (neighbours[0][1] * sr),
                                -1, 1
                            )
                        ) * np.sign( np.cross(neighbours[0][0], sv) )
                    )
            # too few neighbours do not form enough triangles, aka not valid structure
            if (ln := len(neighbours)) < 3:
                continue

            neighbours.sort(key=lambda e: e[2])
            feature: list[tuple[float, float]] = []  # [ (angle, ratio) ]
            for i1 in range(0, ln):
                for i2 in range(i1+1, i1+ln):
                    # here we may cross the 'PI, -PI' boundary, then `later - former`
                    # becomes the clockwise angle (negative value), so we need to
                    # prepare `later` to be greater
                    later, former = Image.iof(neighbours, i2)[2], neighbours[i1][2]
                    if later < former:
                        later += 2 * np.pi
                    angle = later - former
                    # any angle less than PI is ok, but we take 4/5 PI
                    if ((4/5) * np.pi) < angle:
                        break
                    ratio = neighbours[i1][1] / Image.iof(neighbours, i2)[1]
                    feature.append( (angle, ratio) )

            structures.append( (s1[0], np.array(feature)) )

        self.structures = tuple(structures)

    def transform(self, trans_matrix: np.ndarray = None) -> None:
        trans_matrix_ = self.align_trans_matrix if trans_matrix is None else trans_matrix
        self.image = cv.warpPerspective(
            self.image,
            trans_matrix_,
            self.image.shape[:2][::-1],
            flags=cv.INTER_LANCZOS4,
        )

    @staticmethod
    def stretch(
        image: np.ndarray,
        target_dtype: type = np.uint16,
        *,
        extra_factor: float = 1,
    ) -> np.ndarray:
        return (image * (np.iinfo(target_dtype).max / image.max()) * extra_factor).clip(
            np.iinfo(target_dtype).min,
            np.iinfo(target_dtype).max,
        ).astype(target_dtype)

    @staticmethod
    def clip(image: np.ndarray, target_dtype: type = np.uint16) -> np.ndarray:
        return image.clip(
            np.iinfo(target_dtype).min,
            np.iinfo(target_dtype).max,
        ).astype(target_dtype)


class IIO:  # Inter-Image Operation
    @staticmethod
    def score_feature_correlation(
        # the order of `f1` and `f2` does not matter
        f1: np.ndarray,  # f1: SNx2A[ [angle, ratio] ]
        f2: np.ndarray,  # f2: SMx2A[ [angle, ratio] ]
        # TODO: will `weight_cliff_coeff` change from image to image?
        # Can it be somehow auto tuned?
        weight_cliff_coeff: int = 80,
    ) -> float:
        # prepare `f1` and `f2` to be in the 'cartesian product' shape 'S NMx2 A'
        f1e = np.repeat(f1, len(f2), axis=0)
        f2e = np.tile(f2, (len(f1), 1))

        separation = np.linalg.norm(f2e - f1e, axis=1)
        # normalized_separation = 2 / (
        #     1 + np.exp(np.power(weight_cliff_coeff * separation, 3))
        # )
        # # Will the native numpy method be faster?
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            normalized_separation = np.divide(
                2,
                np.add(
                    1,
                    np.exp(
                        np.power(
                            np.multiply(weight_cliff_coeff, separation),
                            3
                        )
                    )
                )
            )

        return normalized_separation.sum() / (len(f1) + len(f2))

    @staticmethod
    def match_star_pairs(
        img1_structures: tuple[tuple[
            np.ndarray,  # S2A[x, y]; structure source star centroid
            np.ndarray,  # SNx2A[ [angle, ratio] ]; feature array of triangles
        ], ...],
        img2_structures: tuple[tuple[np.ndarray, np.ndarray], ...],
        match_correlation_score_threshold: float = 0.3,  # determined from statistic result
    ) -> tuple[tuple[
        np.ndarray, np.ndarray  # img1, img2 star centroid pair
    ], ...]:
        ss2 = list(img2_structures)
        star_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for s1 in img1_structures:
            match_list = []  # [ (structure[centroid, feature], index, score) ]
            for iss2 in range(len(ss2)):
                score = IIO.score_feature_correlation(s1[1], ss2[iss2][1])
                if match_correlation_score_threshold < score:
                    match_list.append( (ss2[iss2], iss2, score) )
            if len(match_list) == 0:
                continue
            # sort `match_list` by score, descending
            match_list.sort(key=lambda e: e[2], reverse=True)
            # select highest score match
            star_pairs.append( (s1[0], match_list[0][0][0]) )
            # remove selected match from search list
            ss2.pop(match_list[0][1])
        if not (4 < len(star_pairs)):
            raise Exception('No more than 4 star pairs matched.')
        return tuple(star_pairs)

    @staticmethod
    def calculate_trans_matrix(
        img1_img2_star_pairs: tuple[tuple[np.ndarray, np.ndarray], ...],
        filter_: bool,
        sample_round: int = None,  # how many sample rounds, `None` for auto
    ) -> np.ndarray:  # S3x3A(m_{ij}) img2 to img1 transformation matrix
        rng = np.random.default_rng()  # random number generator

        # auto determine sample round
        if sample_round is None:
            # TODO: how to find a good sample round?
            sample_round = len(img1_img2_star_pairs)**2
            # # this produce not sufficient sample round
            # probability_of_not_all_coverage = 1 / 10**8
            # sample_round = 1 + int(math.log(
            #     probability_of_not_all_coverage,
            #     1 - 4/len(refe_view_star_pair)
            # ))

        mij: list[np.ndarray] = []  # list[ S9A[m_{ij}] ]
        for i in range(sample_round):
            # sample 4 star pairs to do one round of calculation for matrix
            sample = rng.choice(
                img1_img2_star_pairs,
                size=4,
                replace=False,
                shuffle=False,
            )

            # xr*xv*m31 + xr*yv*m32 + xr*m33 - xv*m11 - yv*m12 - m13 = 0
            # yr*xv*m31 + yr*yv*m32 + yr*m33 - xv*m21 - yv*m22 - m23 = 0
            #
            # [ [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],  X [ m31,  = [ 0,
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1] ]     m32,      0 ]
            #                                                        m33,
            #                                                        m11,
            #                                                        m12,
            #                                                        m13,
            #                                                        m21,
            #                                                        m22,
            #                                                        m23 ]
            #
            # 4 set of pairs and take `m33 = 1`, then we have `Ax=b` as:
            #
            # [ [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],    [ m31,    [ 0,
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1],      m32,      0,
            #   [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],      m33,      0,
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1],      m11,      0,
            #   [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],  X   m12,  =   0,
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1],      m13,      0,
            #   [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],      m21,      0,
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1],      m22,      0,
            #   [    0,     0,  1,   0,   0,  0,   0,   0,  0] ]     m23 ]     1 ]
            #
            # (actually recommend to put the `1` for `m33 = 1` on the last column,
            # otherwise numpy solve result might have float precision problem with
            # `m33`, e.g. fluctuate around `1.0`)
            #
            # ---
            #
            # [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0] == (
            #       [xr, xr, xr, -1, -1, -1, 0, 0, 0]
            #     * [xv, yv,  1, xv, yv,  1, 0, 0, 0]
            # )
            # [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1] == (
            #       [yr, yr, yr, 0, 0, 0, -1, -1, -1]
            #     * [xv, yv,  1, 0, 0, 0, xv, yv,  1]
            # )
            #
            # =>
            #
            # [ [xr*xv, xr*yv, xr, -xv, -yv, -1,   0,   0,  0],
            #   [yr*xv, yr*yv, yr,   0,   0,  0, -xv, -yv, -1] ]
            # ==
            # [ [xr, xr, xr, -1, -1, -1, 0, 0, 0],
            #   [yr, yr, yr, 0, 0, 0, -1, -1, -1] ]
            # *
            # [ [xv, yv, 1, xv, yv, 1,  0,  0, 0],
            #   [xv, yv, 1,  0,  0, 0, xv, yv, 1] ]

            # compose the `A` of `Ax=b`
            A1, A2 = [], []
            for p in sample:
                A1.extend([
                    [ -1, -1, -1,     0, 0, 0,  p[0][0], p[0][0], p[0][0] ],
                    [    0, 0, 0,  -1, -1, -1,  p[0][1], p[0][1], p[0][1] ]
                ])
                A2.extend([
                    [ p[1][0], p[1][1], 1,              0, 0, 0,  p[1][0], p[1][1], 1 ],
                    [             0, 0, 0,  p[1][0], p[1][1], 1,  p[1][0], p[1][1], 1 ]
                ])
            A1, A2 = np.array(A1), np.array(A2)

            # add extra row for `m33=1`
            A = np.vstack(
                (
                    A1 * A2,
                    np.hstack( (np.zeros((1, 8), dtype=int), np.array([[1]])) )
                )
            )

            # compose `b`
            b = np.hstack(
                ( np.zeros(8, dtype=int), np.array([1]) )
            )

            # solve for set of m_{ij} and append the result
            mij.append(np.linalg.solve(A, b))
        mij: np.ndarray = np.array(mij)  # SNx9A[ [m_{ij}] ]; N = sample_round
        # TODO: the distribution of each m_{ij} seems to have two (or multiple?)
        # very close peaks, and they are not symmetrical around the center.
        # Where does it come from??  See the distribution plot:
        # plt.hist(mij.T[i], bins=200)

        # mij might have a lot of outliers, aka large std,
        # we need to filter it before calculate its mean.
        # `vc` for 'validity_criterion'
        vc_max_std = 1
        if filter_:
            vc_min_used_round = sample_round // 3
        else:
            vc_min_used_round = 0

        fmij = mij.view()
        while vc_max_std < fmij.std(axis=0).max():
            # mean and std for each m_{ij} over all sets
            m, s = fmij.mean(axis=0), fmij.std(axis=0)  # S9A[m_{ij}]
            nfmij = fmij[
                np.logical_or(  # SNx9A
                    np.logical_and(  # SNx9A
                        # check for each m_{ij} in each set if it falls within m+-s
                        np.less_equal(m - s, fmij),
                        np.less_equal(fmij, m + s)
                    ),
                    # treat the result of `m33` as always valid, since it might
                    # fluctuate around `1.0` very slightly (when not put at the
                    # last column)
                    np.hstack( (np.full(8, False), np.full(1, True)) )  # S9A
                # only keep sets that have all m_{ij} within m+-s
                ).all(axis=1)  # SNA
            ]
            if (not filter_) and (nfmij.shape[0] < 3):
                break
            fmij = nfmij
        if fmij.shape[0] < vc_min_used_round:
            raise Exception('Can not achieve valid transformation matrix under given criterion.')

        return fmij.mean(axis=0).reshape(3, 3)

    @staticmethod
    def subtract_image(img1: np.ndarray, img2: np.ndarray, factor: float = 1) -> np.ndarray:
        return img1 - factor * img2


class Stack:
    def __init__(self, image_file_path_list: list[str]) -> None:
        ifpl = list(
            e for e in image_file_path_list if not e.endswith(Image.ALIGN_TRANS_MATRIX_FILE_EXT)
        )
        ifpl.sort()
        image_object_list: list[Image] = []
        for p in ifpl:
            image_object_list.append(Image(p))
        self.image_object_tuple: tuple[Image, ...] = tuple(image_object_list)

        self.reference_image_object: Image = None
        self.input_image_object_list: list[Image] = None
        self.set_reference_image(len(self.image_object_tuple) // 2)

    def set_reference_image(self, index: int) -> None:
        self.reference_image_object = self.image_object_tuple[index]
        self.reference_image_object.set_trans_matrix(np.identity(3, np.float64))
        input_image_object_list = list(self.image_object_tuple)
        input_image_object_list.pop(index)
        self.input_image_object_list = input_image_object_list

    @staticmethod
    def _show_progress(progress: int, total: int) -> None:
        percent = progress / total
        bar_length = 50
        left_bar = math.floor(bar_length * percent)
        print(
            'PROGRESS: {:_>{}} / {}  [{}{}{}] {: >5}%'.format(
                progress, len(str(total)),
                total,
                '=' * left_bar,
                '>' if left_bar < bar_length else '',
                ' ' * (bar_length - left_bar - 1),
                int(percent * 1000) / 10,
            ),
            end='\r',
        )

    def align(
        self,
        *,
        save_align_matrix_to_file: bool = True,
        show_progress: bool = True,
        filter_: bool = True,
        max_concurrency: int | None = None,
    ) -> None:
        total = len(self.input_image_object_list)
        if show_progress:
            Stack._show_progress(0, total)
        self.reference_image_object.load(compute_wlred=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.reference_image_object.compute(filter_=filter_)
        self.reference_image_object.release()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_concurrency,
        ) as executor:
            it = executor.map(
                Stack.align_helper,
                (self.reference_image_object,) * total,
                self.input_image_object_list,
                (filter_,) * total,
            )
            for i, result in enumerate(it):
                if result[0] is True:
                    self.input_image_object_list[i].set_trans_matrix(result[1])
                    if save_align_matrix_to_file:
                        self.input_image_object_list[i].save_trans_matrix()
                if show_progress:
                    Stack._show_progress(i + 1, total)
        if show_progress: print()

    @staticmethod
    def align_helper(
        ref: Image,
        sample: Image,
        filter_: bool,
    ) -> tuple[bool, np.ndarray]:
        if sample.align_trans_matrix is not None:
            return (False, sample.align_trans_matrix)
        sample.set_star_structure_neighbour_range(
            ref.star_structure_neighbour_range_low,
            ref.star_structure_neighbour_range_high,
        )
        sample.load(compute_wlred=True)
        sample.compute(filter_=filter_)
        sample.release()
        return (
            True,
            IIO.calculate_trans_matrix(
                IIO.match_star_pairs(
                    ref.structures,
                    sample.structures,
                ),
                filter_=filter_
            ),
        )

    def load_align_matrix_from_file(self) -> None:
        for e in self.image_object_tuple:
            e.load_trans_matrix()

    def reset_align_matrix_in_memory(self) -> None:
        for e in self.image_object_tuple:
            e.align_trans_matrix = None
        self.reference_image_object.set_trans_matrix(np.identity(3, np.float64))

    @enum.unique
    class TYPE(enum.Enum):
        MEAN = enum.auto()
        MEDIAN = enum.auto()
        MEDIAN_OF_MEDIANS = enum.auto()
        MIN = enum.auto()
        MAX = enum.auto()

    def statistics(
        self,
        statistics_type: Stack.TYPE,
        *,
        aligned: bool = True,
        preprocess: Callable[[np.ndarray], np.ndarray] = None,
        memory_budget: float = 2,  # unit is GiB
    ) -> np.ndarray:
        # to obtain the same result as MEDIAN type using MEDIAN_OF_MEDIANS type,
        # set the memory_budget to a very large number so that one batch will
        # cover all the images. (of course you need to have enough memory to
        # hold all the images if doing this)

        self.reference_image_object.load()
        img_shape = self.reference_image_object.image.shape
        img_ori_dtype = self.reference_image_object.image.dtype

        N: int = len(self.image_object_tuple)
        total_size = (  # single image
            64  # float 64
            * img_shape[0] * img_shape[1]  # total pixel
            * 3  # 3 color channels
            / 8  # bit to byte
            / 1024**3  # byte to gigabyte
        ) * N * 3
        # times 3 because (from observation) ndarray takes extra space in memory
        batch_number = math.ceil(total_size / memory_budget)
        img_per_batch = math.ceil(N / batch_number)

        statistics: np.ndarray = None
        statistics_: list[np.ndarray] = None
        # index pool, allow us to shuffle image order when needed
        ip: list[int] = list(range(N))
        match statistics_type:
            case Stack.TYPE.MEAN:
                statistics = np.zeros(img_shape, np.float64)
            case Stack.TYPE.MEDIAN_OF_MEDIANS:
                statistics_ = list()
                random.shuffle(ip)
            case Stack.TYPE.MIN:
                statistics = np.full(
                    img_shape,
                    np.iinfo(img_ori_dtype).max,
                    np.float64,
                )
            case Stack.TYPE.MAX:
                statistics = np.full(
                    img_shape,
                    np.iinfo(img_ori_dtype).min,
                    np.float64,
                )
            case _: raise

        i: int = 0
        while i < N:
            per_batch_list: list[np.ndarray] = list()
            for _ in range(min(img_per_batch, N - i)):
                self.image_object_tuple[ip[i]].load()
                self.image_object_tuple[ip[i]].image = self.image_object_tuple[ip[i]].image.astype(
                    np.float64, casting='safe'
                )
                if preprocess is not None:
                    self.image_object_tuple[ip[i]].image = preprocess(
                        self.image_object_tuple[ip[i]].image
                    )
                if aligned:
                    self.image_object_tuple[ip[i]].transform()
                per_batch_list.append(self.image_object_tuple[ip[i]].image)
                self.image_object_tuple[ip[i]].release()
                i += 1
            per_batch_ndarray: np.ndarray = np.array(per_batch_list)
            match statistics_type:
                case Stack.TYPE.MEAN:
                    statistics += per_batch_ndarray.sum(axis=0) / N
                case Stack.TYPE.MEDIAN_OF_MEDIANS:
                    statistics_.append(np.median(per_batch_ndarray, axis=0))
                case Stack.TYPE.MIN:
                    statistics = np.array(
                        [statistics, per_batch_ndarray.min(axis=0)]
                    ).min(axis=0)
                case Stack.TYPE.MAX:
                    statistics = np.array(
                        [statistics, per_batch_ndarray.max(axis=0)]
                    ).max(axis=0)
                case _: raise
        if statistics_type is Stack.TYPE.MEDIAN_OF_MEDIANS:
            statistics = np.median(np.array(statistics_), axis=0)

        return statistics

    def deprecated_statistics(
        self,
        statistics_types: Stack.TYPE | tuple[Stack.TYPE, ...],
        *,
        aligned: bool = True,
        return_same_dtype: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        self.reference_image_object.load()
        input_dtype = self.reference_image_object.image.dtype

        image_data_list: list[np.ndarray] = list()
        for image_object in self.image_object_tuple:
            image_object.load()
            image_object.image = image_object.image.astype(np.float64, casting='safe')
            if aligned:
                image_object.transform()
            image_data_list.append(image_object.image)
            image_object.release()

        image_data_ndarray: np.ndarray = np.array(image_data_list)

        stats: list[np.ndarray] = []
        single_input: bool = False
        if type(statistics_types) is Stack.TYPE:
            single_input = True
            statistics_types = (statistics_types, )
        for stat_type in statistics_types:
            if stat_type is Stack.TYPE.MEAN:
                stats.append( np.mean(image_data_ndarray, axis=0) )
            elif stat_type is Stack.TYPE.MAX:
                stats.append( np.amax(image_data_ndarray, axis=0) )
            elif stat_type is Stack.TYPE.MEDIAN:
                stats.append( np.median(image_data_ndarray, axis=0) )
            elif stat_type is Stack.TYPE.MIN:
                # If you see ripple pattern artifacts, disable lens distortion
                # correction in the input image
                stats.append( np.amin(image_data_ndarray, axis=0) )
            else:
                raise

        if return_same_dtype:
            if input_dtype == np.float64:
                pass
            elif input_dtype in (np.float16, np.float32):
                for i in range(len(stats)):
                    stats[i] = stats[i].astype(input_dtype)
            else:
                for i in range(len(stats)):
                    stats[i] = stats[i].clip(
                        np.iinfo(input_dtype).min,
                        np.iinfo(input_dtype).max,
                    ).astype(input_dtype)

        if single_input:
            return stats[0]
        return tuple(stats)

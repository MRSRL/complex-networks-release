"""
Created on Thu Aug 16 11:03:49 2018

@author: penglai

routine to generate 2D kt sampling with polynomial (via vdDegree) sampling density
and golden ratio ky-shifting along time
"""

from math import ceil, floor

import numpy as np


def goldenratio_shift(accel, nt):
    GOLDEN_RATIO = 0.618034

    return np.round(np.arange(0, nt) * GOLDEN_RATIO * accel) % accel


def generate_perturbed2dvdkt(
    ny,
    nt,
    accel,
    nCal,
    vdDegree,
    partialFourierFactor=0.0,
    vdFactor=None,
    perturbFactor=0.0,
    adhereFactor=0.0,
):

    vdDegree = max(vdDegree, 0.0)
    perturbFactor = min(max(perturbFactor, 0.0), 1.0)
    adhereFactor = min(max(adhereFactor, 0.0), 1.0)
    nCal = max(nCal, 0)

    if vdFactor == None or vdFactor > accel:
        vdFactor = accel

    yCent = floor(ny / 2.0)
    yRadius = (ny - 1) / 2.0

    if vdDegree > 0:
        vdFactor = vdFactor ** (1.0 / vdDegree)

    accel_aCoef = (vdFactor - 1.0) / vdFactor
    accel_bCoef = 1.0 / vdFactor

    ktMask = np.zeros([ny, nt], np.complex64)
    ktShift = goldenratio_shift(accel, nt)

    for t in range(0, nt):
        # inital sampling with uiform density kt
        ySamp = np.arange(ktShift[t], ny, accel)

        # add random perturbation with certain adherence
        if perturbFactor > 0:
            for n in range(0, ySamp.size):
                if (
                    ySamp[n] < perturbFactor * accel
                    or ySamp[n] >= ny - perturbFactor * accel
                ):
                    continue

                yPerturb = perturbFactor * accel * (np.random.rand() - 0.5)

                ySamp[n] += yPerturb

                if n > 0:
                    ySamp[n - 1] += adhereFactor * yPerturb

                if n < ySamp.size - 1:
                    ySamp[n + 1] += adhereFactor * yPerturb

        ySamp = np.clip(ySamp, 0, ny - 1)

        ySamp = (ySamp - yRadius) / yRadius

        ySamp = ySamp * (accel_aCoef * np.abs(ySamp) + accel_bCoef) ** vdDegree

        ind = np.argsort(np.abs(ySamp))
        ySamp = ySamp[ind]

        yUppHalf = np.where(ySamp >= 0)[0]
        yLowHalf = np.where(ySamp < 0)[0]

        # fit upper half k-space to Cartesian grid
        yAdjFactor = 1.0
        yEdge = floor(ySamp[yUppHalf[0]] * yRadius + yRadius + 0.0001)
        yOffset = 0.0

        for n in range(0, yUppHalf.size):
            # add a very small float 0.0001 to be tolerant to numerical error with floor()
            yLoc = min(
                floor(
                    (yOffset + (ySamp[yUppHalf[n]] - yOffset) * yAdjFactor) * yRadius
                    + yRadius
                    + 0.0001
                ),
                ny - 1,
            )

            if ktMask[yLoc, t] == 0:
                ktMask[yLoc, t] = 1

                yEdge = yLoc + 1

            else:
                ktMask[yEdge, t] = 1
                yOffset = ySamp[yUppHalf[n]]
                yAdjFactor = (yRadius - float(yEdge - yRadius)) / (
                    yRadius * (1 - abs(yOffset))
                )
                yEdge += 1

        # fit lower half k-space to Cartesian grid
        yAdjFactor = 1.0
        yEdge = floor(ySamp[yLowHalf[0]] * yRadius + yRadius + 0.0001)
        yOffset = 0.0

        if ktMask[yEdge, t] == 1:
            yEdge -= 1
            yOffset = ySamp[yLowHalf[0]]
            yAdjFactor = (yRadius + float(yEdge - yRadius)) / (
                yRadius * (1.0 - abs(yOffset))
            )

        for n in range(0, yLowHalf.size):
            yLoc = max(
                floor(
                    (yOffset + (ySamp[yLowHalf[n]] - yOffset) * yAdjFactor) * yRadius
                    + yRadius
                    + 0.0001
                ),
                0,
            )

            if ktMask[yLoc, t] == 0:
                ktMask[yLoc, t] = 1

                yEdge = yLoc + 1

            else:
                ktMask[yEdge, t] = 1
                yOffset = ySamp[yLowHalf[n]]
                yAdjFactor = (yRadius - float(yEdge - yRadius)) / (
                    yRadius * (1 - abs(yOffset))
                )
                yEdge -= 1

    # at last, add calibration data
    ktMask[(yCent - ceil(nCal / 2)) : (yCent + nCal - 1 - ceil(nCal / 2)), :] = 1

    # CMS: simulate partial Fourier scheme with alternating ky lines
    if partialFourierFactor > 0.0:
        nyMask = int(ny * partialFourierFactor)
        # print(nyMask)
        # print(ny-nyMask)
        ktMask[(ny - nyMask) : ny, 0::2] = 0
        ktMask[0:nyMask, 1::2] = 0

    return ktMask

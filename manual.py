"""
manual.py -- EVERYTHING written with plain loops. No built-in operations.

NOT used anywhere in this file:
    cv2.equalizeHist, cv2.filter2D, cv2.medianBlur, cv2.calcHist, cv2.cvtColor
    np.cumsum, np.median, np.sum, np.pad, np.sort, np.argmin, np.max, np.min
    np.fft.fft2, np.fft.fftshift, np.convolve, np.histogram, np.where, np.clip

ONLY used:
    cv2.imread / cv2.imwrite   -> to open and save the file (not an operation)
    matplotlib                 -> to draw the picture on screen
    np.zeros                   -> to create an empty array to fill in
    math.log, math.exp, math.sqrt, math.cos, math.sin, math.pi
    random.random              -> only for generating test noise

Run:  python manual.py
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


# =======================================================================
# PART 0 -- basic helpers (these replace the numpy functions)
# =======================================================================

def read(path):
    """Open a grayscale image. If the file is missing, build a test image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    # fallback test image, built by hand
    size = 96
    img = np.zeros((size, size), np.uint8)
    for i in range(size):
        for j in range(size):
            v = 40 + 80 * j / size                       # dark ramp
            if (i - 30) ** 2 + (j - 30) ** 2 < 250:      # bright circle
                v = 200
            if 55 < i < 80 and 55 < j < 85:              # gray box
                v = 130
            img[i][j] = int(v)
    return img


def size_of(img):
    """height, width. Written out instead of using img.shape."""
    return len(img), len(img[0])


def empty(h, w, kind=float):
    """Create an all-zero array of the right size, ready to fill in."""
    return np.zeros((h, w), kind)


def clip(v):
    """Force one number into 0..255. Replaces np.clip."""
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)


def to_uint8(arr):
    """Clip a whole float array into a displayable image. Replaces np.clip + astype."""
    h, w = size_of(arr)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = clip(arr[i][j])
    return out


def find_min_max(arr):
    """Smallest and largest value. Replaces arr.min() and arr.max()."""
    h, w = size_of(arr)
    mn = arr[0][0]
    mx = arr[0][0]
    for i in range(h):
        for j in range(w):
            if arr[i][j] < mn:
                mn = arr[i][j]
            if arr[i][j] > mx:
                mx = arr[i][j]
    return mn, mx


def normalize(arr):
    """
    Stretch any range into 0..255.
    USE THIS for Sobel / Laplacian / spectrum (they contain negative numbers).
    """
    h, w = size_of(arr)
    mn, mx = find_min_max(arr)
    span = mx - mn
    if span == 0:
        span = 1
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = clip((arr[i][j] - mn) / span * 255)
    return out


def show(images, titles, cols=4):
    """Draw the images in a row. Only used for display."""
    n = len(images)
    if cols > n:
        cols = n
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(3.6 * cols, 3.8 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =======================================================================
# PART 1 -- POINT PROCESSING (one pixel in, one pixel out)
# =======================================================================

def negative(img):
    """s = 255 - r"""
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = 255 - img[i][j]
    return out


def log_transform(img):
    """
    s = c * log(1 + r)      brightens dark areas a lot, bright areas a little
    c is chosen so that the brightest pixel lands exactly on 255.
    """
    h, w = size_of(img)
    mn, mx = find_min_max(img)
    c = 255.0 / math.log(1.0 + mx)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = clip(c * math.log(1.0 + img[i][j]))
    return out


def gamma_transform(img, gamma):
    """
    s = 255 * (r/255) ^ gamma
    gamma < 1 -> brighter,  gamma > 1 -> darker
    """
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            r = img[i][j] / 255.0
            out[i][j] = clip(255.0 * (r ** gamma))
    return out


def threshold(img, T):
    """s = 255 if r >= T else 0"""
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            if img[i][j] >= T:
                out[i][j] = 255
            else:
                out[i][j] = 0
    return out


def bit_planes(img):
    """
    Split the 8 bits of every pixel into 8 images.
    (value >> k) & 1  keeps only bit number k.
    plane 7 = most important (the picture), plane 0 = least (noise).
    """
    h, w = size_of(img)
    planes = []
    for k in range(8):
        plane = empty(h, w, np.uint8)
        for i in range(h):
            for j in range(w):
                bit = (int(img[i][j]) >> k) & 1
                plane[i][j] = 255 * bit
        planes.append(plane)
    return planes


# =======================================================================
# PART 2 -- HISTOGRAM
# =======================================================================

def histogram(img):
    """
    h[k] = how many pixels have brightness k.
    The pixel value IS the index of the counter.
    """
    h, w = size_of(img)
    hist = [0] * 256                  # 256 counters, all starting at zero
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist


def pdf_of(img):
    """PDF = histogram divided by the total number of pixels."""
    h, w = size_of(img)
    total = h * w
    hist = histogram(img)
    p = [0.0] * 256
    for k in range(256):
        p[k] = hist[k] / total
    return p


def cdf_of(img):
    """
    CDF = running total of the PDF.  Replaces np.cumsum.
    c[k] = fraction of pixels that are k or darker.  It ends at 1.0.
    """
    p = pdf_of(img)
    c = [0.0] * 256
    running = 0.0
    for k in range(256):
        running = running + p[k]
        c[k] = running
    return c


def equalize(img):
    """
    HISTOGRAM EQUALISATION
        s[r] = round(255 * CDF(r))       <- the mapping table
        output[i][j] = s[ input[i][j] ]  <- just look it up
    """
    c = cdf_of(img)

    s = [0] * 256                                   # the mapping table
    for r in range(256):
        s[r] = int(round(255.0 * c[r]))

    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = s[img[i][j]]                # table lookup
    return out


def match_histogram(source, reference):
    """
    HISTOGRAM MATCHING
    For each old value r, find the value z where the two CDFs are closest.
    Replaces np.argmin with a plain "smallest so far" loop.
    """
    c1 = cdf_of(source)
    c2 = cdf_of(reference)

    table = [0] * 256
    for r in range(256):
        best_z = 0
        best_diff = abs(c2[0] - c1[r])
        for z in range(256):
            diff = abs(c2[z] - c1[r])
            if diff < best_diff:                    # found a closer one
                best_diff = diff
                best_z = z
        table[r] = best_z

    h, w = size_of(source)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = table[source[i][j]]
    return out


# =======================================================================
# PART 3 -- CONVOLUTION (a pixel and its neighbours)
# =======================================================================

def pad(img, p):
    """
    Add a border of p pixels by copying the edge. Replaces np.pad.
    Needed because at the border the kernel hangs off the image.
    """
    h, w = size_of(img)
    out = empty(h + 2 * p, w + 2 * p, float)
    for i in range(h + 2 * p):
        for j in range(w + 2 * p):
            si = i - p                     # where this comes from in the original
            sj = j - p
            if si < 0:
                si = 0                     # above the top -> use the first row
            if si > h - 1:
                si = h - 1                 # below the bottom -> use the last row
            if sj < 0:
                sj = 0
            if sj > w - 1:
                sj = w - 1
            out[i][j] = img[si][sj]
    return out


def flip_kernel(k):
    """Rotate the kernel 180 degrees. This is what makes it CONVOLUTION."""
    ks = len(k)
    out = empty(ks, ks, float)
    for m in range(ks):
        for n in range(ks):
            out[m][n] = k[ks - 1 - m][ks - 1 - n]
    return out


def convolve(img, k, flip=True):
    """
    CONVOLUTION from scratch.
        put the kernel centre on pixel (i,j)
        multiply each kernel number by the image number under it
        add all of them up -> that is output pixel (i,j)

    flip=True  -> convolution
    flip=False -> correlation (what cv2.filter2D really does)

    Returns FLOAT. Use to_uint8() for blurs, normalize() for edges.
    """
    if flip:
        k = flip_kernel(k)

    ks = len(k)
    p = ks // 2
    padded = pad(img, p)

    h, w = size_of(img)
    out = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            total = 0.0
            for m in range(ks):                 # walk over the kernel
                for n in range(ks):
                    total = total + padded[i + m][j + n] * k[m][n]
            out[i][j] = total
    return out


def box_kernel(size):
    """Averaging kernel. Every value is 1/(size*size) so the total is 1."""
    k = empty(size, size, float)
    for m in range(size):
        for n in range(size):
            k[m][n] = 1.0 / (size * size)
    return k


def gaussian_kernel(size, sigma):
    """
    w = exp( -(s^2 + t^2) / (2 sigma^2) ), then divided by its total.
    The centre counts most, far neighbours count least.
    """
    k = empty(size, size, float)
    c = size // 2
    total = 0.0
    for m in range(size):
        for n in range(size):
            s = m - c
            t = n - c
            v = math.exp(-(s * s + t * t) / (2.0 * sigma * sigma))
            k[m][n] = v
            total = total + v
    for m in range(size):                       # normalise so it sums to 1
        for n in range(size):
            k[m][n] = k[m][n] / total
    return k


def make_kernel(rows):
    """Turn a list of lists into a kernel array."""
    ks = len(rows)
    k = empty(ks, ks, float)
    for m in range(ks):
        for n in range(ks):
            k[m][n] = rows[m][n]
    return k


SOBEL_X = make_kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = make_kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACIAN = make_kernel([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
SHARPEN = make_kernel([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def gradient_magnitude(gx, gy):
    """magnitude = sqrt(gx^2 + gy^2)   -- the length of the edge arrow."""
    h, w = size_of(gx)
    out = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            out[i][j] = math.sqrt(gx[i][j] * gx[i][j] + gy[i][j] * gy[i][j])
    return out


def unsharp_mask(img, k_strength, size=5, sigma=1.0):
    """
    1. blur the image      -> loses the fine detail
    2. mask = original - blurred  -> ONLY the fine detail
    3. output = original + k * mask   -> detail counted twice = sharper
    """
    blurred = convolve(img, gaussian_kernel(size, sigma))

    h, w = size_of(img)
    mask = empty(h, w, float)
    out = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            mask[i][j] = img[i][j] - blurred[i][j]
            out[i][j] = img[i][j] + k_strength * mask[i][j]
    return to_uint8(out), normalize(mask), to_uint8(blurred)


# =======================================================================
# PART 4 -- MEDIAN AND ORDER FILTERS (not convolution)
# =======================================================================

def sort_list(values):
    """
    Insertion sort. Replaces np.sort / sorted().
    Take each value and slide it left until it is in the right place.
    """
    for i in range(1, len(values)):
        key = values[i]
        j = i - 1
        while j >= 0 and values[j] > key:
            values[j + 1] = values[j]
            j = j - 1
        values[j + 1] = key
    return values


def median_filter(img, size=3):
    """
    Best filter for salt-and-pepper noise.
    Take the neighbourhood, SORT it, keep the MIDDLE value.
    A single white dot ends up at the end of the sorted list, so it is
    never chosen -- the dot disappears and the edges stay sharp.
    """
    p = size // 2
    padded = pad(img, p)
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    middle = (size * size) // 2

    for i in range(h):
        for j in range(w):
            values = []
            for m in range(size):
                for n in range(size):
                    values.append(padded[i + m][j + n])
            values = sort_list(values)
            out[i][j] = clip(values[middle])
    return out


def order_filter(img, size=3, kind="min"):
    """kind = "min" (removes white dots) or "max" (removes black dots)."""
    p = size // 2
    padded = pad(img, p)
    h, w = size_of(img)
    out = empty(h, w, np.uint8)

    for i in range(h):
        for j in range(w):
            best = padded[i][j]
            for m in range(size):
                for n in range(size):
                    v = padded[i + m][j + n]
                    if kind == "min" and v < best:
                        best = v
                    if kind == "max" and v > best:
                        best = v
            out[i][j] = clip(best)
    return out


# =======================================================================
# PART 5 -- NOISE (for testing)
# =======================================================================

def add_salt_pepper(img, amount=0.06):
    """Random black and white dots."""
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            r = random.random()
            if r < amount / 2:
                out[i][j] = 255
            elif r < amount:
                out[i][j] = 0
            else:
                out[i][j] = img[i][j]
    return out


def add_gaussian_noise(img, sigma=20):
    """Grainy noise everywhere."""
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = clip(img[i][j] + random.gauss(0, sigma))
    return out


def add_periodic_noise(img, A=45, u0=8, v0=8):
    """
    n(x,y) = A * sin( 2 pi (u0 x / M + v0 y / N) )
    Makes diagonal stripes -- the thing a notch filter removes.
    """
    h, w = size_of(img)
    out = empty(h, w, np.uint8)
    for i in range(h):
        for j in range(w):
            n = A * math.sin(2 * math.pi * (u0 * i / h + v0 * j / w))
            out[i][j] = clip(img[i][j] + n)
    return out


# =======================================================================
# PART 6 -- FOURIER TRANSFORM, written out by hand
# =======================================================================

def dft_1d(values, inverse=False):
    """
    One-dimensional DFT.
        F[k] = sum over n of  f[n] * ( cos(a) + j sin(a) ),  a = -2 pi k n / N

    Complex numbers are kept as two separate real numbers (re and im), so
    no complex maths library is needed. Multiplying (a+jb)(c+jd) gives
        real = a*c - b*d ,  imag = a*d + b*c
    """
    N = len(values)
    out_re = [0.0] * N
    out_im = [0.0] * N
    sign = 1.0 if inverse else -1.0

    for k in range(N):
        sum_re = 0.0
        sum_im = 0.0
        for n in range(N):
            angle = sign * 2.0 * math.pi * k * n / N
            c = math.cos(angle)
            s = math.sin(angle)
            a = values[n][0]        # real part of the input
            b = values[n][1]        # imaginary part of the input
            sum_re = sum_re + a * c - b * s
            sum_im = sum_im + a * s + b * c
        out_re[k] = sum_re
        out_im[k] = sum_im

    result = []
    for k in range(N):
        result.append([out_re[k], out_im[k]])
    return result


def dft_2d(img, inverse=False):
    """
    Two-dimensional DFT, done as: transform every ROW, then every COLUMN.
    (Doing it directly would need 4 nested loops and take forever.)

    The image is stored as a list of rows, each pixel being [real, imag].
    KEEP THE IMAGE SMALL -- 64x64 is instant, 128x128 takes a few seconds.
    """
    h, w = size_of(img) if isinstance(img, np.ndarray) else (len(img), len(img[0]))

    # step 0: put the image into [real, imag] form
    data = []
    for i in range(h):
        row = []
        for j in range(w):
            v = img[i][j]
            if isinstance(v, list):
                row.append([v[0], v[1]])
            else:
                row.append([float(v), 0.0])
        data.append(row)

    # step 1: transform each row
    for i in range(h):
        data[i] = dft_1d(data[i], inverse)

    # step 2: transform each column
    for j in range(w):
        column = []
        for i in range(h):
            column.append(data[i][j])
        column = dft_1d(column, inverse)
        for i in range(h):
            data[i][j] = column[i]

    # step 3: the inverse needs dividing by the number of pixels
    if inverse:
        for i in range(h):
            for j in range(w):
                data[i][j][0] = data[i][j][0] / (h * w)
                data[i][j][1] = data[i][j][1] / (h * w)
    return data


def shift(F):
    """
    Move the centre of the spectrum into the middle of the picture.
    Replaces np.fft.fftshift. It swaps the four quarters diagonally.
    Running it twice gets you back to the start.
    """
    h = len(F)
    w = len(F[0])
    out = []
    for i in range(h):
        out.append([[0.0, 0.0]] * w)
    out = [[[0.0, 0.0] for _ in range(w)] for _ in range(h)]

    for i in range(h):
        for j in range(w):
            ni = (i + h // 2) % h
            nj = (j + w // 2) % w
            out[ni][nj] = F[i][j]
    return out


def spectrum(F):
    """
    The picture of the spectrum: log(1 + magnitude).
        magnitude = sqrt(real^2 + imag^2)
    The log is needed because the centre value is millions of times bigger
    than everything else.
    """
    h = len(F)
    w = len(F[0])
    mag = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            re = F[i][j][0]
            im = F[i][j][1]
            mag[i][j] = math.log(1.0 + math.sqrt(re * re + im * im))
    return normalize(mag)


def real_image(F):
    """Throw away the imaginary part and turn the result back into an image."""
    h = len(F)
    w = len(F[0])
    out = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            out[i][j] = F[i][j][0]
    return to_uint8(out)


def distance(h, w, i, j, ci=None, cj=None):
    """Distance from point (i,j) to the centre (or to any chosen point)."""
    if ci is None:
        ci = h // 2
    if cj is None:
        cj = w // 2
    di = i - ci
    dj = j - cj
    return math.sqrt(di * di + dj * dj)


def ideal_lowpass(h, w, D0):
    """H = 1 inside the circle of radius D0, 0 outside. Causes RINGING."""
    H = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            if distance(h, w, i, j) <= D0:
                H[i][j] = 1.0
            else:
                H[i][j] = 0.0
    return H


def gaussian_lowpass(h, w, D0):
    """H = exp(-D^2 / (2 D0^2)). Fades out smoothly, so NEVER rings."""
    H = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            D = distance(h, w, i, j)
            H[i][j] = math.exp(-(D * D) / (2.0 * D0 * D0))
    return H


def butterworth_lowpass(h, w, D0, n=2):
    """H = 1 / (1 + (D/D0)^(2n)). H is exactly 0.5 when D = D0."""
    H = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            D = distance(h, w, i, j)
            H[i][j] = 1.0 / (1.0 + (D / D0) ** (2 * n))
    return H


def to_highpass(H):
    """Any highpass filter = 1 - the lowpass filter."""
    h, w = size_of(H)
    out = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            out[i][j] = 1.0 - H[i][j]
    return out


def notch_reject(h, w, points, D0=8, n=2):
    """
    NOTCH REJECT FILTER -- removes periodic noise (stripes).

    Stripes show up as bright dots in the spectrum, always in a symmetric
    pair. We punch a small hole at each dot AND at its mirror on the other
    side of the centre. The holes are MULTIPLIED together.

        one hole:  1 / (1 + (D0/D)^(2n))
                   this is 0 right at the dot and rises to 1 further away
    """
    H = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            H[i][j] = 1.0                       # start by keeping everything

    for (u0, v0) in points:
        du = u0 - h // 2                        # how far the dot is from centre
        dv = v0 - w // 2
        for s in (1, -1):                       # the dot, then its mirror
            ci = h // 2 + s * du
            cj = w // 2 + s * dv
            for i in range(h):
                for j in range(w):
                    D = distance(h, w, i, j, ci, cj)
                    if D < 0.000001:
                        D = 0.000001            # never divide by zero
                    hole = 1.0 / (1.0 + (D0 / D) ** (2 * n))
                    H[i][j] = H[i][j] * hole
    return H


def apply_filter(img, H):
    """
    The standard steps:
        1. F = DFT(image)
        2. centre the spectrum
        3. G = H * F          (multiply, point by point)
        4. un-centre
        5. image = inverse DFT(G), keep the real part
    Returns output, centred F, centred G.
    """
    F = shift(dft_2d(img))                       # steps 1 and 2

    h = len(F)
    w = len(F[0])
    G = [[[0.0, 0.0] for _ in range(w)] for _ in range(h)]
    for i in range(h):                           # step 3
        for j in range(w):
            G[i][j] = [F[i][j][0] * H[i][j], F[i][j][1] * H[i][j]]

    back = dft_2d(shift(G), inverse=True)        # steps 4 and 5
    return real_image(back), F, G


def find_spikes(F, how_many=2, skip=6):
    """
    Find the brightest dots in the spectrum, ignoring the centre.
    In the exam you can also just LOOK at the spectrum plot and read the
    coordinates off with your mouse.
    """
    h = len(F)
    w = len(F[0])

    mag = empty(h, w, float)
    for i in range(h):
        for j in range(w):
            if distance(h, w, i, j) < skip:
                mag[i][j] = 0.0                  # ignore the bright centre
            else:
                re = F[i][j][0]
                im = F[i][j][1]
                mag[i][j] = math.sqrt(re * re + im * im)

    points = []
    for _ in range(how_many):
        bi, bj, best = 0, 0, -1.0
        for i in range(h):
            for j in range(w):
                if mag[i][j] > best:
                    best = mag[i][j]
                    bi, bj = i, j
        points.append((bi, bj))
        for i in range(bi - 3, bi + 4):          # blank it out so the next
            for j in range(bj - 3, bj + 4):      # search finds a different dot
                if 0 <= i < h and 0 <= j < w:
                    mag[i][j] = 0.0
    return points


def psnr(a, b):
    """Image quality score. Higher is better. Above 30 dB is good."""
    h, w = size_of(a)
    total = 0.0
    for i in range(h):
        for j in range(w):
            d = float(a[i][j]) - float(b[i][j])
            total = total + d * d
    mse = total / (h * w)
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)


# =======================================================================
# DEMO -- runs every function
# =======================================================================

if __name__ == "__main__":
    img = read("input.jpg")
    print("image size:", size_of(img))

    # --- point processing
    show([img, negative(img), log_transform(img), gamma_transform(img, 0.4)],
         ["Original", "Negative", "Log", "Gamma 0.4"])

    show([img, threshold(img, 128)] + bit_planes(img)[5:],
         ["Original", "Threshold", "Bit plane 5", "Bit plane 6", "Bit plane 7"],
         cols=5)

    # --- histogram
    eq = equalize(img)
    plt.figure(figsize=(11, 6))
    plt.subplot(2, 2, 1); plt.imshow(img, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(2, 2, 2); plt.bar(range(256), histogram(img)); plt.title("Histogram before")
    plt.subplot(2, 2, 3); plt.imshow(eq, cmap="gray"); plt.title("Equalised"); plt.axis("off")
    plt.subplot(2, 2, 4); plt.bar(range(256), histogram(eq)); plt.title("Histogram after")
    plt.tight_layout(); plt.show()

    # --- convolution
    blur = to_uint8(convolve(img, box_kernel(3)))
    sharp = to_uint8(convolve(img, SHARPEN, flip=False))
    gx = convolve(img, SOBEL_X, flip=False)
    gy = convolve(img, SOBEL_Y, flip=False)
    mag = normalize(gradient_magnitude(gx, gy))
    show([img, blur, sharp, normalize(gx), mag],
         ["Original", "Box blur", "Sharpen", "Sobel X", "Gradient magnitude"], cols=5)

    us, mask, blurred = unsharp_mask(img, 1.5)
    show([img, blurred, mask, us],
         ["Original", "Blurred", "Mask = f - blur", "Unsharp result"])

    # --- noise
    noisy = add_salt_pepper(img, 0.07)
    show([img, noisy, to_uint8(convolve(noisy, box_kernel(3))), median_filter(noisy)],
         ["Original", "Salt & pepper", "Blur (bad)", "Median (good)"])

    # --- frequency domain (kept small because the hand-written DFT is slow)
    small = empty(64, 64, np.uint8)
    for i in range(64):
        for j in range(64):
            small[i][j] = img[i * len(img) // 64][j * len(img[0]) // 64]

    striped = add_periodic_noise(small, 45, 8, 8)
    F = shift(dft_2d(striped))
    pts = find_spikes(F, 2, skip=6)
    print("noise spikes at:", pts)

    # D0 is the size of the hole. Too small = stripes remain, too big = blurry.
    # Small image -> small D0. Try 3, 5, 10 and keep whichever looks best.
    H = notch_reject(64, 64, pts, D0=3, n=2)
    out, F, G = apply_filter(striped, H)

    show([striped, spectrum(F), normalize(H), spectrum(G), out],
         ["Striped noise", "Spectrum", "Notch filter H", "Filtered", "Restored"],
         cols=5)
    print("PSNR: noisy %.1f dB -> restored %.1f dB" % (psnr(small, striped),
                                                       psnr(small, out)))

    Hlp = gaussian_lowpass(64, 64, 12)
    lp, _, _ = apply_filter(small, Hlp)
    hp, _, _ = apply_filter(small, to_highpass(Hlp))
    show([small, normalize(Hlp), lp, hp],
         ["Original", "Gaussian LP filter", "Lowpass (blur)", "Highpass (edges)"])

    print("done")

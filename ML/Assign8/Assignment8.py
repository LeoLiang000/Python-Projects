import numpy as np
from PIL import Image


# convolve input array to kernel
def convolve(input_array, kernel):
    # get kernel height and width
    k_h, k_w = kernel.shape[0], kernel.shape[1]

    # add padding to keep output the same size as input
    pad_img = np.pad(input_array, (((k_h - 1) // 2, (k_h - 1) // 2), ((k_w - 1) // 2, (k_w - 1) // 2)))

    # compute convolution
    ret = np.zeros((input_array.shape[0], input_array.shape[1]))  # store result
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            temp_convolution = (kernel * pad_img[i: i + k_h, j: j + k_w]).sum()
            ret[i, j] = temp_convolution

    return ret


# convolve image input with kernels
def convolution(img):
    img = Image.open(img).resize((200, 200))
    img_rgb = np.array(img.convert('RGB'))
    img_r = img_rgb[:, :, 0]  # single channel

    kernel1 = np.array([[
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    ]])

    kernel2 = np.array([[
        0, -1, 0,
        -1, 8, -1,
        0, -1, 0
    ]])

    Image.fromarray(np.uint8(convolve(img_r, kernel1))).show()
    Image.fromarray(np.uint8(convolve(img_r, kernel2))).show()


def main():
    convolution('image1.jpg')


if __name__ == '__main__':
    main()

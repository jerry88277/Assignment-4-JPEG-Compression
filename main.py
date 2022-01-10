# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:33:55 2021

@author: RE6104019
"""

import os
import cv2
import math
import array
import struct
import numpy as np    
import pandas as pd

from PIL import Image
from tqdm import tqdm
from scipy.fft import dct
from scipy.signal import convolve2d
from multiprocessing.pool import Pool

# In[]

class Encoder(object):
    def __init__(self, image_address, N1 = 8, N2 = 8):
        self.address = image_address
        self.N1 = N1
        self.N2 = N2

        self.raw_image = []
        self.height = 0
        self.width = 0

        self.seg_file = []
        self.seg_number = 0

        self.dct_res = []
        self.quant_table = []
        
        self.quant_res = []


    #######################################
    #This is a image reading funx
    #-------------------------------------#
    #INPUT :     address of the image
    #OUTPUT :    raw_image, height, width
    #raw_image : image data saved in list
    #height:     height of the image
    #width:      width of the image
    #######################################

    def image_reading(self, image_address): # image_address = image_path
        im = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
        res = np.array(im)
        (height, width) = res.shape
        height = height - height % 8
        width = width - width % 8
        raw_image = res[0:height][0:width]
        # im.close()
        return raw_image, height, width
    
    
    #######################################
    #This funx is in order to seg image in to pieces
    #Each pieces' size is N1 * N2
    #-------------------------------------#
    #INPUT:        image, height, width, N1, N2
    #image:        the input data
    #height:       height of the image
    #width:        width of the image
    #N1:           row number of each piece
    #N2:           col number of each piece
    #-------------------------------------#
    #OUTPUT:       res, l
    #res:          the data after processing
    #l:            the piece's number
    ########################################

    def seg(self, image, height, width, N1, N2):
        res = []
        row_begin = 0
        while row_begin < height:
            col_begin = 0
            while col_begin < width:
                #print(col_begin)
                res.append(image[row_begin:(row_begin + N1), col_begin:(col_begin + N2)])
                col_begin += N2
            row_begin += N1
        return res, len(res)
    #######################################
    #This funx implements DCT
    #-------------------------------------#
    #INPUT: input_data, length,N1, N2
    #input_data: the length of input_data list is length, each element is a N1*N2 matrics
    #N1: row number of each element
    #N2: col number of each element
    #OUTPUT: res
    #res: new data after DCT
    #######################################
    def DCT(self, input_data, seg_number, N1, N2):
        PI = math.pi
        res = []
        # get matrix C
        c = np.zeros((N1, N2), dtype = np.float64)
        for i in range(N1):
            for j in range(N2):
                if i == 0:
                    c_u = math.sqrt(1/N1)
                else:
                    c_u = math.sqrt(2/N1)
                if j == 0:
                    c_v = math.sqrt(1/N2)
                else:
                    c_v = math.sqrt(2/N2)
                c[i][j] = c_u * c_v
        
        cos_coff = []
        for u in range(N1):
            for v in range(N2):
                cos_a = np.array(list(math.cos((i + 0.5) * PI / N1 * u) for i in range(N1)))
                cos_b = np.array(list(math.cos((i + 0.5) * PI / N2 * v) for i in range(N2)))
                cos_a = cos_a.reshape(N1, 1)
                cos_coff.append(cos_a * cos_b)

        for index in range(seg_number):
            temp = np.zeros((N1, N2), dtype = np.float64)
            for u in range(N1):
                for v in range(N2):
                    temp[u][v] = np.sum(input_data[index] * cos_coff[u * N1 + v]) * c[u][v]
                    #temp[u][v] = temp[u][v].astype(np.int)
            res.append(temp)
        return res
    
    def quantification(self, input_data, len_input_data, quant_table):
        res = []
        for i in range(len_input_data):
            if isinstance(input_data[i], list):
                dividend = np.array(input_data[i])
            else:
                dividend = input_data[i]
            temp = (dividend/quant_table).astype(np.int32)
            res.append(temp)
        return res

    def quant_table_reading(self, address):
        f = open(address, 'r')
        table = f.read()
        f.close()
        res = []
        temp = []
        current_number = 0
        for char in table:
            if char == '\n':
                temp.append(current_number)
                res.append(temp)
                current_number = 0
                temp = []
            elif char == ' ':
                temp.append(current_number)
                current_number = 0
            else:
                current_number = current_number * 10 + int(char)
        temp.append(current_number)
        res.append(temp)
        return np.array(res)

    def huffman_build_tree(self, input_data, length, N1, N2):
        d = {}
        for i in range(length):
            for n_1 in range(N1):
                for n_2 in range(N2):
                    if d.get(input_data[i][n_1][n_2], -1) == -1:
                        d[input_data[i][n_1][n_2]] = 1
                    else:
                        d[input_data[i][n_1][n_2]] += 1
        class node(object):
            def __init__(self,key,value,left, right, flag):
                self.key = key
                self.value = value
                self.left = left
                self.right = right
                self.flag = flag

        table = []
        for i in d.keys():
            temp = node(d[i], i, -1, -1, False)
            table.append(temp)
        
        table = sorted(table, key = lambda x : x.key)
        while len(table) != 1:
            temp = node(table[0].key + table[1].key, -1, table[0], table[1], False)
            table.pop(0)
            table.pop(0)
            index = 0
            for i in table:
                if i.key < temp.key:
                    index += 1
                else:
                    break
            
            table.insert(index, temp)
        return table[0]
    
    def generate_huffman_table(self, root):
        st = [root]
        current_key = ''
        huffman_table = {}
        while len(st) > 0:
            #print(st[-1].key)
            if st[-1].left == -1:
                huffman_table[st[-1].value] = current_key
                st[-1].flag = True
                st.pop()
                #print(len(st))
            else:
                if st[-1].left.flag == False:
                    #print("left key is %d" % st[-1].left.key)
                    st[-1].key = current_key
                    #print("now key is")
                    #print(current_key)
                    current_key += '1'
                    st.append(st[-1].left)
                    
                elif st[-1].right.flag == False:
                    #print("right, key is")
                    #print(st[-1].key)
                    current_key = st[-1].key
                    current_key += '0'
                    st.append(st[-1].right)
                    
                else:
                    st[-1].flag = True
                    st.pop()
        
        return huffman_table

    def generate_compressed_file(self, huffman_table, quant_res, seg_number, N1, N2):
        trans_codes = ''
        for i in range(seg_number):
            for n_1 in range(N1):
                for n_2 in range(N2):
                    trans_codes += huffman_table[quant_res[i][n_1][n_2]]
        #f = open('compress_lena', 'wb')
        #f.write(trans_codes)
        #f.close()
        return trans_codes, len(trans_codes)

    def start(self):
        self.raw_image, self.height, self.width = self.image_reading(self.address)
        
        print("image reading finished")
        print("the height of image is %d, the width of image is %d" % (self.height, self.width))

        self.seg_file, self.seg_number = self.seg(self.raw_image, self.height, self.width, self.N1, self.N2)       
        print("image seg finished")
        print("the number of pieces is %d" %self.seg_number)

        self.dct_res = self.DCT(self.seg_file, self.seg_number, self.N1, self.N2)
        print("DCT finished")

        quant_table_address = 'quant_table.txt'
        self.quant_table = self.quant_table_reading(quant_table_address)
        print("quant_table loading finished")

        self.quant_res = self.quantification(self.dct_res, self.seg_number, self.quant_table)
        print("quantification finished")

        huffman_tree_root = self.huffman_build_tree(self.quant_res, self.seg_number, self.N1, self.N2)
        print("huffman_tree building finished")

        huffman_table = self.generate_huffman_table(huffman_tree_root)
        print("huffman table generating finished")

        trans_codes, bit_number = self.generate_compressed_file(huffman_table, self.quant_res,self.seg_number, self.N1, self.N2)
        print("transcode_generate finished, the final bit number is %d" % bit_number)

        return trans_codes, huffman_tree_root
    
    def start_420(self):
        self.raw_image, self.height, self.width = self.image_reading(self.address)
        
        print("image reading finished")
        print("the height of image is %d, the width of image is %d" % (self.height, self.width))

        self.seg_file, self.seg_number = self.seg(self.raw_image, self.height, self.width, self.N1, self.N2)       
        print("image seg finished")
        print("the number of pieces is %d" %self.seg_number)

        self.dct_res = self.DCT(self.seg_file, self.seg_number, self.N1, self.N2)
        print("DCT finished")

        quant_table_address = 'quant_table.txt'
        self.quant_table = self.quant_table_reading(quant_table_address)
        print("quant_table loading finished")

        self.quant_res = self.quantification(self.dct_res, self.seg_number, self.quant_table)
        print("quantification finished")

        huffman_tree_root = self.huffman_build_tree(self.quant_res, self.seg_number, self.N1, self.N2)
        print("huffman_tree building finished")

        huffman_table = self.generate_huffman_table(huffman_tree_root)
        print("huffman table generating finished")

        trans_codes, bit_number = self.generate_compressed_file(huffman_table, self.quant_res,self.seg_number, self.N1, self.N2)
        print("transcode_generate finished, the final bit number is %d" % bit_number)

        return trans_codes, huffman_tree_root

class Decoder(object):
    def __init__(self, compress_file, huffman_root, quant_table, seg_number, N1, N2, height, width):
        self.height = height
        self.width = width
        self.seg_number = seg_number
        self.N1 = N1
        self.N2 = N2
        self.huffman_root = huffman_root
        self.compress_file = compress_file
        
        self.quant_table = quant_table

        self.huffman_decoder_res = []
        self.quant_decoder_res = []
        self.IDCT_res = []
        self.restruction_res = []
        
    
    def huffman_decoder(self, input_data, huffman_root, seg_number, N1, N2):
        current_node = huffman_root
        h_decoder_res = []
        l = len(input_data)
        index = 0
        while index < l:
            if current_node.left == -1:
                h_decoder_res.append(current_node.value)
                current_node = huffman_root
            elif input_data[index] == '1':
                current_node = current_node.left
                index += 1
            else:
                current_node = current_node.right
                index += 1
        h_decoder_res.append(current_node.value)
        output_block = np.array(h_decoder_res).reshape((seg_number, N1, N2))
        return output_block


    def decoder_quant_res(self, input_data, seg_number,quant_table):
        res = []
        for i in range(seg_number):
            temp = input_data[i] * quant_table
            res.append(temp)
        return res

    def IDCT(self, input_data, seg_number, N1, N2):
        PI = math.pi
        res = []
        # get matrix C
        c = np.zeros((N1, N2), dtype = np.float64)
        for i in range(N1):
            for j in range(N2):
                if i == 0:
                    c_u = math.sqrt(1/N1)
                else:
                    c_u = math.sqrt(2/N1)
                if j == 0:
                    c_v = math.sqrt(1/N2)
                else:
                    c_v = math.sqrt(2/N2)
                c[i][j] = c_u * c_v
        
        cos_coff = []
        for i in range(N1):
            for j in range(N2):
                cos_a = np.array(list(math.cos((i + 0.5) * PI / N1 * u) for u in range(N1)))
                cos_b = np.array(list(math.cos((j + 0.5) * PI / N2 * v) for v in range(N2)))
                cos_a = cos_a.reshape(N1, 1)
                cos_coff.append(cos_a * cos_b)

        for index in range(seg_number):
            temp = np.zeros((N1, N2), dtype = np.float64)
            for i in range(N1):
                for j in range(N2):
                    temp[i][j] = np.sum(input_data[index] * cos_coff[i * N1 + j] * c)
                    temp[i][j] = temp[i][j].astype(np.uint8)
            res.append(temp)
        return res

    def combine(self, input_data, seg_number, N1, N2, height, width):
        res = np.zeros((height, width))
        row_begin = 0
        index = 0
        while row_begin < height:
            col_begin = 0
            while col_begin < width:
                res[row_begin:(row_begin + N1), col_begin:(col_begin + N2)] = input_data[index]
                index += 1
                col_begin += N2
            row_begin += N1
        return res
    
    def combine_color(self, input_data, seg_number, N1, N2, height, width, channel):
        res = np.zeros((height, width, channel))
        channel_begin = 0
        row_begin = 0
        index = 0
        while channel_begin < channel:
            while row_begin < height:
                col_begin = 0
                while col_begin < width:
                    res[row_begin:(row_begin + N1), col_begin:(col_begin + N2), channel] = input_data[index]
                    index += 1
                    col_begin += N2
                row_begin += N1
            channel_begin += 1
        return res
    
    def start(self):
        self.huffman_decoder_res = self.huffman_decoder(self.compress_file, self.huffman_root, self.seg_number, self.N1, self.N2)
        self.quant_decoder_res = self.decoder_quant_res(self.huffman_decoder_res,self.seg_number,self.quant_table)
        self.IDCT_res = self.IDCT(self.quant_decoder_res, self.seg_number, self.N1, self.N2)
        self.restruction_res = self.combine(self.IDCT_res, self.seg_number, self.N1, self.N2, self.height, self.width)
        return 1
    
class Decoder_420(object):
    def __init__(self, compress_file, huffman_root, seg_number, N1, N2):
        self.seg_number = seg_number
        self.N1 = N1
        self.N2 = N2
        self.huffman_root = huffman_root
        self.compress_file = compress_file

        self.huffman_decoder_res = []
    
    def huffman_decoder(self, input_data, huffman_root, seg_number, N1, N2):
        current_node = huffman_root
        h_decoder_res = []
        l = len(input_data)
        index = 0
        while index < l:
            if current_node.left == -1:
                h_decoder_res.append(current_node.value)
                current_node = huffman_root
            elif input_data[index] == '1':
                current_node = current_node.left
                index += 1
            else:
                current_node = current_node.right
                index += 1
        h_decoder_res.append(current_node.value)
        output_block = np.array(h_decoder_res).reshape((seg_number, N1, N2))
        return output_block

def RMSE(original_image, compressed_image): # compressed_image = rgb_img_compressed
    
    if len(original_image.shape) == 2:
        
        MSE = np.sum((original_image - compressed_image)) / len(original_image)**2
    
    elif len(original_image.shape) == 3:
        MSE = []
        for i_ch in range(len(original_image.shape)):
            i_MSE = np.sum((original_image[:,:, i_ch] - compressed_image[:,:, i_ch])) / len(original_image)**2    
            MSE.append(i_MSE)
        MSE = np.mean(np.sum(np.array(MSE)))
    
    RMSE_score = np.sqrt(MSE)
    
    return RMSE_score

def PSNR(original_image, compressed_image):
    
    if len(original_image.shape) == 2:
        
        MSE = np.sum((original_image - compressed_image)) / len(original_image)**2
    
    elif len(original_image.shape) == 3:
        MSE = []
        for i_ch in range(len(original_image.shape)):
            i_MSE = np.sum((original_image[:,:, i_ch] - compressed_image[:,:, i_ch])) / len(original_image)**2    
            MSE.append(i_MSE)
        MSE = np.mean(np.sum(np.array(MSE)))
    
    PSNR_score = 20 * np.log10(np.max(original_image) / np.sqrt(MSE))
    
    return PSNR_score

def image_RGB2YCrCb(original_image):
    
    R = original_image[:, :, 2]
    G = original_image[:, :, 1]
    B = original_image[:, :, 0]
    
    Y = 0.299 * R + 0.578 * G + 0.114 * B
    # Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    Cb = 0.565 * (B - Y) 
    # Cr = 0.500 * R - 0.419 * G - 0.081 * B + 128
    Cr = 0.713 * (R - Y)
    
    image_YCrCb = np.dstack((Y, Cb, Cr))
    
    return image_YCrCb

class Downsampling():
    def __init__(self, ratio='4:2:0'):
        assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
        self.ratio = ratio
        
    def __call__(self, x):
        # No subsampling
        if self.ratio == '4:4:4':
            return x
        else:
            # Downsample with a window of 2 in the horizontal direction
            if self.ratio == '4:2:2':
                kernel = np.array([[0.5], [0.5]])
                out = np.repeat(convolve2d(x, kernel, mode='valid')[::2,:], 2, axis=0)
            # Downsample with a window of 2 in both directions
            else:
                kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
                out = np.repeat(np.repeat(convolve2d(x, kernel, mode='valid')[::2,::2], 2, axis=0), 2, axis=1)
            return np.round(out).astype('int')

class ImageBlock():
    def __init__(self, block_height=8, block_width=8):
        self.block_height = block_height
        self.block_width = block_width
        self.left_padding = self.right_padding = self.top_padding = self.bottom_padding = 0
    
    def forward(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_channel = image.shape[2]
    
        # Vertical padding
        if self.image_height % self.block_height != 0:
            vpad = self.image_height % self.block_height
            self.top_padding = vpad // 2 
            self.bottom_padding = vpad - self.top_padding
            image = np.concatenate((np.repeat(image[:1], self.top_padding, 0), image, 
                                    np.repeat(image[-1:], self.bottom_padding, 0)), axis=0)
            
        # Horizontal padding
        if self.image_width % self.block_width != 0:
            hpad = self.image_width % self.block_width
            self.left_padding = hpad // 2 
            self.right_padding = hpad - self.left_padding
            image = np.concatenate((np.repeat(image[:,:1], self.left_padding, 1), image, 
                                    np.repeat(image[:,-1:], self.right_padding, 1)), axis=1)
    
        # Update dimension
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

        # Create blocks
        blocks = []
        indices = []
        for i in range(0, self.image_height, self.block_height):
            for j in range(0, self.image_width, self.block_width):
                for k in range(self.image_channel):
                    blocks.append(image[i:i+self.block_height, j:j+self.block_width, k])
                    indices.append((i,j,k))
                    
        blocks = np.array(blocks)
        indices = np.array(indices)
        return blocks, indices
    
    def backward(self, blocks, indices):
        # Empty image array
        image = np.zeros((self.image_height, self.image_width, self.image_channel)).astype(int)
        for block, index in zip(blocks, indices):
            i, j, k = index
            image[i:i+self.block_height, j:j+self.block_width, k] = block
            
        # Remove padding
        if self.top_padding > 0:
            image = image[self.top_padding:,:,:]
        if self.bottom_padding > 0:
            image = image[:-self.bottom_padding,:,:] 
        if self.left_padding > 0:
            image = image[:,self.left_padding:,:]
        if self.right_padding > 0:
            image = image[:,:-self.right_padding,:]
        return image

class DCT2D():
    def __init__(self, norm='ortho'):
        if norm is not None:
            assert norm == 'ortho', "norm needs to be in {None, 'ortho'}"
        self.norm = norm
    
    def forward(self, x):
        out = dct(dct(x, norm=self.norm, axis=0), norm=self.norm, axis=1)
        return out
    
    def backward(self,x):
        out = dct(dct(x, type=3, norm=self.norm, axis=0), type=3, norm=self.norm, axis=1)
        return np.round(out)

class Quantization():
    # Qunatiztion matrices
    # https://www.impulseadventure.com/photo/jpeg-quantization.html
    
    # Luminance
    Q_lum = np.array([[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]])
    # Chrominance
    Q_chr = np.array([[17,18,24,47,99,99,99,99],
                      [18,21,26,66,99,99,99,99],
                      [24,26,56,99,99,99,99,99],
                      [47,66,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99]])
    
    def forward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')
        
        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = np.round(x/Q)
        return out
    
    def backward(self, x, channel_type):
        assert channel_type in ('lum', 'chr')
        
        if channel_type == 'lum':
            Q = self.Q_lum
        else:
            Q = self.Q_chr

        out = x*Q
        return out

def process_block(block, index): # block = blocks, index = indices
    # DCT
    encoded = dct2d.forward(block)
    if index[2] == 0:
        channel_type = 'lum'
    else:
        channel_type = 'chr'
        
    # Quantization
    encoded = quantization.forward(encoded, channel_type)
    
    # Dequantization
    decoded = quantization.backward(encoded, channel_type)
    
    # Reverse DCT
    compressed = dct2d.backward(decoded)
    return compressed



# In[]

data_folder = 'data'

save_path = 'output/task1'

if not os.path.exists(save_path):
    os.makedirs(save_path)

task1_list = os.listdir(data_folder)
task1_list = [i for i in task1_list if 'color' not in i ]

# score_table = pd.DataFrame(columns = ['Image name', 'Level','RMSE', 'PSNR'])
score_list = []

for i_image_name in tqdm(task1_list):
    
    image_path = os.path.join(data_folder, i_image_name)
    image_encoder = Encoder(image_path)
    
    trans_codes, huffman_tree_root = image_encoder.start()
    
    file_name = f'{i_image_name}_bitstream.txt'
    with open(os.path.join(save_path, file_name), 'w') as text_file:
        text_file.write(trans_codes)
        text_file.close()
    
    compress_decoder = Decoder(trans_codes,
                               huffman_tree_root,
                               image_encoder.quant_table,
                               image_encoder.seg_number,
                               image_encoder.N1,
                               image_encoder.N2,
                               image_encoder.height,
                               image_encoder.width)
    
    compress_decoder.start()
    compressed_image = compress_decoder.restruction_res
    
    image_save_path = os.path.join(save_path , f'{i_image_name}_DCT_compressed.png')
    cv2.imwrite(image_save_path, compressed_image)

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    RMSE_score = RMSE(original_image, compressed_image)
    PSNR_score = PSNR(original_image, compressed_image)

    score_list.append([i_image_name, 'gray', RMSE_score, PSNR_score])

score_table = pd.DataFrame(score_list, columns = ['Image name', 'Level','RMSE', 'PSNR'])


# In[]

data_folder = 'data'

save_path = 'output/task2'

if not os.path.exists(save_path):
    os.makedirs(save_path)

task2_list = os.listdir(data_folder)
task2_list = [i for i in task2_list if 'color' in i ]


for i_image_name in tqdm(task2_list):
    
    image_path = os.path.join(data_folder, i_image_name)
    original_image = cv2.imread(image_path)

    lum_downsample = Downsampling(ratio='4:4:4')
    chr_downsample = Downsampling(ratio='4:2:0')
    image_block = ImageBlock(block_height=8, block_width=8)
    dct2d = DCT2D(norm='ortho')
    quantization = Quantization()
    
    image_YCrCb = image_RGB2YCrCb(original_image)

    ycc_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2YCrCb)
    ycc_img = ycc_img.astype(int)-128
    
    Y = lum_downsample(ycc_img[:, :, 0])
    Cr = chr_downsample(ycc_img[:, :, 1])
    Cb = chr_downsample(ycc_img[:, :, 2])
    ycc_img = np.stack((Y, Cr, Cb), axis = 2)
    
    blocks, indices = image_block.forward(ycc_img)
    # compressed = np.array(Pool().starmap(process_block, zip(blocks, indices)))
    compressed = []
    for index, value in tqdm(enumerate(zip(blocks, indices))):
        i_block = value[0]
        i_index = value[-1]
        i_compressed = process_block(i_block, i_index)
        compressed.append(i_compressed)
        
    compressed = np.array(compressed)
    
    ## transform to huffmancode
    image_encoder_420 = Encoder(image_path)
    
    huffman_tree_root = image_encoder_420.huffman_build_tree(compressed, compressed.shape[0], compressed.shape[1], compressed.shape[2])
    print("huffman_tree building finished")

    huffman_table = image_encoder_420.generate_huffman_table(huffman_tree_root)
    print("huffman table generating finished")

    trans_codes, bit_number = image_encoder_420.generate_compressed_file(huffman_table, compressed, compressed.shape[0], compressed.shape[1], compressed.shape[2])
    print("transcode_generate finished, the final bit number is %d" % bit_number)
    
    file_name = f'{i_image_name}_bitstream.txt'
    with open(os.path.join(save_path, file_name), 'w') as text_file:
        text_file.write(trans_codes)
        text_file.close()
    
    ## re-transform from huffmancode
    bit_Decoder_420 = Decoder_420(trans_codes, huffman_tree_root, compressed.shape[0], compressed.shape[1], compressed.shape[2])
    huffman_decoder_res = bit_Decoder_420.huffman_decoder(trans_codes, huffman_tree_root, compressed.shape[0], compressed.shape[1], compressed.shape[2])
    
    
    # ycc_img_compressed = image_block.backward(compressed, indices)
    ycc_img_compressed = image_block.backward(huffman_decoder_res, indices)
    ycc_img_compressed = (ycc_img_compressed+128).astype('uint8')
    rgb_img_compressed = cv2.cvtColor(ycc_img_compressed, cv2.COLOR_YCrCb2RGB)
        
    image_save_path = os.path.join(save_path , f'{i_image_name}_420compressed2.png')
    cv2.imwrite(image_save_path, rgb_img_compressed)
    
    RMSE_score = RMSE(original_image, rgb_img_compressed)
    PSNR_score = PSNR(original_image, rgb_img_compressed)

    score_list.append([i_image_name, 'color', RMSE_score, PSNR_score])
    
score_table = pd.DataFrame(score_list, columns = ['Image name', 'Level','RMSE', 'PSNR'])
    
# In[]

score_table.to_csv('score_table.csv')


# In[]

import matplotlib.pyplot as plt

qt = pd.read_csv(r'D:\NCKU\Course\Digital Image Processing And Computer vision\HW4\JPEG Compression\quant_table.txt',
                 sep = ' ')

fig, ax = plt.subplots(figsize = (12, 8))
ax.matshow(qt, cmap=plt.cm.Blues)
plt.xticks(size = 12)
plt.xticks(size = 12)
for i in range(len(qt) + 1):
    for j in range(len(qt)):
        c = qt.iloc[j, i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.savefig('qt.png')
plt.close()





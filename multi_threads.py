import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImageGenerator:
    def __init__(self,pic_dir,corp_size=(20,20), filter_size=20):
        """配置参数"""
        # 图片后缀名，防止文件夹中存在非图片文件
        self.suffix = ['jpg', 'jpeg', 'JPG', 'JPEG', 'gif', 'GIF', 'png', 'PNG']
        # 保存子图RGB平均值，用于后期通过颜色寻找相似子图
        self.means = {}
        # 每个子图的指纹，指纹包含图片颜色结构信息
        self.codes = {}
        # 数组形式保存子图，用于后期拼接母图
        self.pic_dic = {}
        # 记录图像直方图信息
        self.hist_dic = {}
        # 填充子图分辨率
        self.corp_size = corp_size
        # 指纹信息像素
        self.new_corp_size=(8,8)
        # rgb最接近的个数，再比较指纹
        self.filter_size = filter_size
        # 子图（微信好友头像）所在文件夹，
        self.sub_pic_dir = pic_dir

        # 初始化操作
        # 加载图像列表，计算rgb平均值，指纹信息
        self.load_imgs()
        
    def compute_new_image_size(self):
        '''根据原图宽高，裁剪小区块宽高，计算出新图像的宽高'''
        self.OldImage = Image.open(self.pic_path)
        # 如果原图是RGBA，换成RGB
        if self.extension=='.png':
            self.OldImage=self.OldImage.convert('RGB')
            self.save_name=self.save_name+'.jpg'
        width, height = self.OldImage.size
        print(f'原图宽高为{width, height}')
        # 拼接之后图片的‘宽’,拼接之后图片的‘高’
        self.to_width = width // self.corp_size[0] * self.corp_size[0]
        self.to_height = height // self.corp_size[1] * self.corp_size[1]
        print(f'重绘后宽高为{self.to_width, self.to_height}')
        self.w_times, self.h_times = self.to_width // self.corp_size[0],self.to_height // self.corp_size[1]
        # 将原图resize成计算出来的分辨率，Image.LANCZOS 是一种插值算法，用于图像的缩放操作
        self.picture = self.OldImage.resize((int(self.to_width), int(self.to_height)), Image.LANCZOS)
        # 将图片转为数组格式
        self.picture = np.array(self.picture)
        # 生成一个空数组
        self.output_img = np.zeros_like(self.picture)
        
    def run(self,pic_path,save_name=None):
        # 保存文件名
        self.save_name = save_name
        # 母图路径
        self.pic_path = pic_path

        filename, self.extension = os.path.splitext(os.path.basename(pic_path))
        if not save_name:
            save_name = filename + '-generated' + self.extension
        if not os.path.exists(save_name):
            os.path.join(os.path.dirname(__file__), 'output')
        self.save_name = os.path.join(os.path.dirname(__file__), 'output', save_name)
        
        # 根据图片计算信息
        self.compute_new_image_size()
        self.merge_image()
        self.output_image()
        
    def load_imgs(self):
        # 获取文件夹下所有文件
        self.pic_list = os.listdir(self.sub_pic_dir)
        self.item_num = len(self.pic_list)
        self.compute_image()
        
    def rgb_mean(self,rgb_pic):
        """计算RGB通道图片平均值"""
        r_mean = np.mean(rgb_pic[:, :, 0])
        g_mean = np.mean(rgb_pic[:, :, 1])
        b_mean = np.mean(rgb_pic[:, :, 2])
        val = np.array([r_mean, g_mean, b_mean])
        return val
    
    def pic_code(self,image: np.ndarray):
        """生成子图的指纹信息，指纹信息中包含图片颜色结构"""
        width, height = image.shape
        avg = image.mean()
        fingerprint = np.array([1 if image[i, j] > avg else 0 for i in range(width) for j in range(height)])
        return fingerprint

    # 增加多线程实现
    def compute_image(self):
        '''计算出小图像的RGB平均值和指纹'''
        error_num = 0
        
        def process_image(pic):
            nonlocal error_num
            if pic.split('.')[-1] in self.suffix:
                path = os.path.join(self.sub_pic_dir, pic)
                try:
                    img = Image.open(path).convert('RGB').resize(self.corp_size, Image.LANCZOS)
                    idx = self.pic_list.index(pic)  # Get the index of the current image
                    self.codes[idx] = self.pic_code(np.array(img.convert('L').resize(self.new_corp_size, Image.LANCZOS)))
                    self.means[idx] = self.rgb_mean(np.array(img))
                    self.pic_dic[idx] = np.array(img)
                    self.hist_dic[idx] = img.histogram()
                except OSError:
                    error_num += 1
        
        with ThreadPoolExecutor() as executor:
            # Use the map method to apply the process_image function to each image in pic_list
            executor.map(process_image, self.pic_list)
            
        print(f'小图像加载完成, {self.item_num - error_num:4}加载成功数量，加载失败数量{error_num:2}')
        
    def structure_similarity(self, section, candidate):
        """从候选图片选取结构最相似的子图"""
        section = Image.fromarray(section).convert('L')
        one_hot = self.pic_code(np.array(section.resize(self.new_corp_size, Image.LANCZOS)))
        candidate = [(key_, np.equal(one_hot, self.codes[key_]).mean()) for key_, _ in candidate]
        most_similar = max(candidate, key=lambda item: item[1])
        return self.pic_dic[most_similar[0]]
    
    def color_similarity(self, pic_slice, top_n):
        """计算图片与所有子图RGB均值的欧式距离，返回最相似的候选子图列表"""
        slice_mean = self.rgb_mean(pic_slice)
        diff_list = [(key_, np.linalg.norm(slice_mean - value_)) for key_, value_ in self.means.items()]
        filter_ = sorted(diff_list, key=lambda item: item[1])[:top_n]
        return filter_
    
    # 增加多线程实现
    def merge_image(self):
        '''合并出新图像'''
        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(self.w_times), desc='☺️合并拼接生成图像中...'):
                for j in range(self.h_times):
                    section = self.picture[j * self.corp_size[1]:(j + 1) * self.corp_size[1],
                            i * self.corp_size[0]:(i + 1) * self.corp_size[0], :]
                    candidate = self.color_similarity(section, top_n=self.filter_size)
                    most_similar = self.structure_similarity(section, candidate)
                    self.output_img[j * self.corp_size[1]:(j + 1) * self.corp_size[1], 
                                    i * self.corp_size[0]:(i + 1) * self.corp_size[0], :] = most_similar

    def output_image(self):
        '''将数组转为图片'''
        # Check if the output directory exists, if not, create it
        output_dir = os.path.dirname(self.save_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.output_img = Image.fromarray(self.output_img)
        self.output_img.save(self.save_name)

generator=ImageGenerator(pic_dir='../images',corp_size=(20, 20))
generator.run(pic_path='./imgs/dog.JPG',save_name='output.jpg')

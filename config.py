# train
epoch = 1000
num_classes = 124
batch_size = 10
device = 'cpu'  # cpu or 'cuda:0'
train_image_path = './data/train/'  # 每个类别一个文件夹, 类别使用数字
valid_image_path = './data/train/'  # 每个类别一个文件夹, 类别使用数字
num_workers = 8  # 加载数据集线程并发数
best_loss = 0.01  # 当loss小于等于该值会保存模型
save_model_iter = 500  # 每多少次保存一份模型
model_output_dir = './data/resnet_cls/'
resume = False  # 是否从断点处开始训练
chkpt = './data/resnet_cls/best_11.pth'  # 断点训练的模型
lr = 0.003

# predict
predict_model = './data/resnet_cls/best_4.pth'
predict_image_path = './data/train/'  # 每个类别一个文件夹, 类别使用数字

image_format = 'jpg'

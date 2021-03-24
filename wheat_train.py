import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import albumentations as A
from effdet import get_efficientdet_config, EfficientDet
from effdet.loss import DetectionLoss
from effdet.anchors import Anchors, AnchorLabeler
from tqdm.notebook import tqdm as tqdm
from effdet.bench import DetBenchPredict
import matplotlib.pyplot as plt
from albumentations.pytorch import transforms
from albumentations import Compose
import time


# small 600MB dataset
def read_dataframe(csv_file_dir, ratio):
    '''take in directory of train.csv and split images according to some ratio into train, val, test dataframe'''
    df = pd.read_csv(csv_file_dir)
    df.head()
    df = df.sample(frac=1)  # shuffle dataframe

    # get list of image id
    unique_names = df["image_id"].unique()
    np.random.shuffle(unique_names)  # randomly shuffle list of image names

    name_train = unique_names[:round(len(unique_names) * ratio[0])]
    name_val = unique_names[round(len(unique_names) * ratio[0]):round(len(unique_names) * (ratio[0] + ratio[1]))]
    name_test = unique_names[round(len(unique_names) * (ratio[0] + ratio[1])):]

    # print("number of images in each data split")
    # print("train:", len(name_train))
    # print("validation:", len(name_val))
    # print("test:", len(name_test))

    # print("\n")

    # print("number of boxes in each data split")
    df_train = df[df["image_id"].isin(name_train)]
    df_val = df[df["image_id"].isin(name_val)]
    df_test = df[df["image_id"].isin(name_test)]
    # print("train:", len(df_train))
    # print("validation:", len(df_val))
    # print("test:", len(df_test))
    return df_train, df_val, df_test


# dataset object

# SOURCE: this code  is inspired, with some copying from https://www.kaggle.com/ottiliemitchell/wheat-project


###Run this function only once to check the validity of all datapoints we have access to
def check_image_sizes(df_train, df_val, df_test, train_dir, bs=16):
    # convert to Dataset objects

    # TODO: might need to insert some transformation
    train_data = GetDataset(df_train, train_dir)
    val_data = GetDataset(df_val, train_dir)
    test_data = GetDataset(df_test, train_dir)

    # check if shapes in all datasets are consistent
    old_shape = train_data[0][0].shape
    for i in range(len(train_data)):
        if train_data[i][0].shape != old_shape:
            raise Exception()
    print('checked train data')

    for i in range(len(val_data)):
        if val_data[i][0].shape != old_shape:
            raise Exception()
    print('checked val data')

    for i in range(len(test_data)):
        if test_data[i][0].shape != old_shape:
            raise Exception()
    print('checked test data')
    return True


def get_model_name(train_batch, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format("efficientDet",
                                                   train_batch,
                                                   learning_rate,
                                                   epoch)
    return path


def collate_fn(batch):
    # return tuple(zip(*batch))
    return zip(*batch)
    # return batch


from torch.hub import load_state_dict_from_url


def get_net():
    # config = get_efficientdet_config('tf_efficientdet_d4')
    config = get_efficientdet_config('tf_efficientdet_d0')
    print(config)
    net = EfficientDet(config, pretrained_backbone=False)

    count = 0
    for param in net.parameters():
        count += torch.prod(torch.tensor(param.shape))
    print(count)
    # checkpoint = torch.load('/content/drive/MyDrive/efficientdet-pytorch/efficientdet_d4-5b370b7a.pth')
    # checkpoint = torch.load('/content/drive/MyDrive/efficientdet-pytorch/efficientdet_d0-d92fd44f.pth')
    # net.load_state_dict(checkpoint)
    state_dict = load_state_dict_from_url(config.url, progress=False, map_location='cpu')
    net.load_state_dict(state_dict, strict=True)

    net.reset_head(num_classes=1)

    return net


class ToFeatures(object):
    """Convert ndarrays in sample to Features."""

    def __call__(self, **kwargs):
        with torch.no_grad():
            kwargs.update({'image': EffDet.fpn(EffDet.backbone(kwargs['image'].unsqueeze(0)))})

        if 'mask' in kwargs.keys():
            kwargs.update({'mask': mask_to_tensor(kwargs['mask'], self.num_classes, sigmoid=self.sigmoid)})
        return kwargs


class GetFeatureDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = dataframe["image_id"].unique()

        # decode box dimension and calculate corner coordinates

        x_l = []
        x_h = []
        y_l = []
        y_h = []
        for item in dataframe["bbox"]:
            x_l.append(float(item.strip('][').split(', ')[0]))
            y_l.append(float(item.strip('][').split(', ')[1]))
            x_h.append(x_l[-1] + float(item.strip('][').split(', ')[2]))
            y_h.append(y_l[-1] + float(item.strip('][').split(', ')[3]))
        self.dataframe["x_l"] = x_l
        self.dataframe["y_l"] = y_l
        self.dataframe["x_h"] = x_h
        self.dataframe["y_h"] = y_h

    def __getitem__(self, idx):
        # Load images and details
        image_id = self.image_ids[idx]
        details = self.dataframe[self.dataframe["image_id"] == image_id]
        img_path = os.path.join(self.image_dir, image_id) + ".jpg"
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Row of Dataframe of a particular index.
        boxes = details[['x_l', 'y_l', 'x_h', 'y_h']].values

        # To find area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Convert it into tensor dataType
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((details.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((details.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transform(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.long)
        return image, target  # , image_id

    def __len__(self) -> int:
        return len(self.image_ids)


def get_Dataset(dframe, train_dir):
    # convert to Dataset objects

    composed_transform = Compose(
        [A.augmentations.transforms.Cutout(num_holes=8, max_h_size=64, max_w_size=64, p=0.5), transforms.ToTensor()])
    # to extract features, uncomment this line and comment the line prior
    # composed_transform = Compose([transforms.ToTensor(), ToFeatures()])

    dset = GetFeatureDataset(dframe, train_dir, composed_transform)
    return dset


def get_Loader(dset, batch_size=16):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dloader


def check_validation(model, val_loader):
    pbar = tqdm(val_loader, desc='validating')
    total_loss = 0
    total_class_loss = 0
    total_box_loss = 0
    total_count = 0
    for item1, targets in pbar:
        if use_Cuda and torch.cuda.is_available():
            images = torch.stack(item1).cuda()
            class_target = [item['labels'] for item in targets]
            box_target = [item['boxes'].float() for item in targets]

            cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                box_target,
                class_target)

            cls_targets_cuda = [item.cuda() for item in cls_targets]
            box_targets_cuda = [item.cuda() for item in box_targets]

        with torch.no_grad():
            class_out, box_out = EffDet(images)

            loss, class_loss, box_loss = loss_fn(class_out, box_out, cls_targets_cuda, box_targets_cuda, num_positives)
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_box_loss += box_loss.item()
            total_count = total_count + len(targets)
    return total_loss / total_count, total_class_loss / total_count, total_box_loss / total_count

train_bs = 8
val_bs = 8

csv_file_dir = "/home/administrator/wheat/smaller_dataset/global-wheat-detection/train.csv"
train_dir = "/home/administrator/wheat/smaller_dataset/global-wheat-detection/train"

state_dir = "/home/administrator/wheat/state"
plot_dir = "/home/administrator/wheat/plot"

df_train, df_val, df_test = read_dataframe(csv_file_dir, [0.8,0.1,0.1])
train_dset = get_Dataset(df_train, train_dir)
train_loader = get_Loader(train_dset, batch_size = train_bs)

val_dset = get_Dataset(df_val, train_dir)
val_loader = get_Loader(val_dset, batch_size = val_bs)





################################################################TRAINING CODE####################################
use_Cuda = True

EffDet = get_net()
# freeze certain parts of the network
for param in EffDet.backbone.parameters():
    param.requires_grad = False
for param in EffDet.fpn.parameters():
    param.requires_grad = False

if use_Cuda and torch.cuda.is_available():
    EffDet.cuda()

# set up Anchor and Anchor labellers
config = EffDet.config
anchors = Anchors(config.min_level, config.max_level,
                  config.num_scales, config.aspect_ratios,
                  config.anchor_scale, [1024, 1024])  # ben: this is my hack to get around the 128 dimension problem
anchors_cuda = Anchors(config.min_level, config.max_level,
                       config.num_scales, config.aspect_ratios,
                       config.anchor_scale,
                       [1024, 1024])  # ben: this is my hack to get around the 128 dimension problem
anchor_labeler = AnchorLabeler(anchors, EffDet.config['num_classes'], match_threshold=0.5)

# set up loss function
loss_fn = DetectionLoss(EffDet.config)
loss_fn.use_jit = True

# set up optimizer
learning_rate = 1e-3
num_epoch = 100
optimizer = torch.optim.Adam(EffDet.parameters(), lr=learning_rate)

# set up reference image for printing
image = val_dset[0][0]
numpy_image_org = image.permute(1, 2, 0).numpy()

test_img = image.unsqueeze(0)
img_info = {}
img_info['img_scale'] = None
img_info['img_size'] = torch.tensor([1024, 1024]).cuda()

if use_Cuda and torch.cuda.is_available():
    test_img = test_img.cuda()

train_loss_step = []
train_class_loss_step = []
train_box_loss_step = []

train_loss_epoch = []
train_class_loss_epoch = []
train_box_loss_epoch = []

val_loss_epoch = []
val_class_loss_epoch = []
val_box_loss_epoch = []

quarter_epoch = round(len(train_loader)/4)


for epoch in range(num_epoch):
    epoch_start = time.time()
    train_loss_total = 0
    train_class_loss_total = 0
    train_box_loss_total = 0
    num_datapoints = 0


    print("training epoch:", epoch)
    for iter, (item1, targets) in enumerate(train_loader):
        if use_Cuda and torch.cuda.is_available():
            images = torch.stack(item1).cuda()
            class_target = [item['labels'] for item in targets]
            box_target = [item['boxes'].float() for item in targets]

            cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                box_target,
                class_target)

            cls_targets_cuda = [item.cuda() for item in cls_targets]
            box_targets_cuda = [item.cuda() for item in box_targets]

        # print(images.shape)
        class_out, box_out = EffDet(images)

        # print(class_out[0].shape)
        # print(cls_targets[0].shape)
        # print(box_targets[0].shape)

        loss, class_loss, box_loss = loss_fn(class_out, box_out, cls_targets_cuda, box_targets_cuda, num_positives)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_class_loss_total += class_loss.item()
        train_box_loss_total += box_loss.item()
        train_loss_total += loss.item()

        train_loss_step.append(loss.item())
        train_class_loss_step.append(class_loss.item())
        train_box_loss_step.append(box_loss.item())

        num_datapoints += len(targets)

        if (iter + 1) % quarter_epoch == 0:
            print('loss:', train_loss_total / num_datapoints, 'class loss:', train_class_loss_total / num_datapoints,
                  'box loss:', train_box_loss_total / num_datapoints)

    train_loss_epoch.append(train_loss_total / num_datapoints)
    train_class_loss_epoch.append(train_class_loss_total / num_datapoints)
    train_box_loss_epoch.append(train_box_loss_total / num_datapoints)

    print('validating')
    val_loss, val_class_loss, val_box_loss = check_validation(EffDet, val_loader)
    val_loss_epoch.append(val_loss)
    val_class_loss_epoch.append(val_class_loss)
    val_box_loss_epoch.append(val_box_loss)

    print("overall epoch:", epoch, "train loss:", train_loss_epoch[-1], "val loss:", val_loss_epoch[-1])
    print("class epoch:", epoch, "train loss:", train_class_loss_epoch[-1], "val loss:", val_class_loss_epoch[-1])
    print("box epoch:", epoch, "train loss:", train_box_loss_epoch[-1], "val loss:", val_box_loss_epoch[-1])

    # plot reference image:

    model = DetBenchPredict(EffDet)
    model.anchors = anchors_cuda
    if use_Cuda and torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        outputs = (model(test_img, img_info=img_info)[0])

    numpy_image = np.copy(numpy_image_org)
    for box in outputs:
        cv2.rectangle(numpy_image, (box[1].int(), box[0].int()), (box[3].int(), box[2].int()), (0, 1, 0), 2)
    plt.pause(0.001)
    plt.imsave(os.path.join(plot_dir, "epoch{}.png".format(epoch)), numpy_image)

    # save state:
    model_path = get_model_name(train_bs, learning_rate, epoch)

    torch.save(EffDet.state_dict(), os.path.join(state_dir,model_path))

    # epoch
    print("epoch: ", epoch)
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 3, 1)
    plt.plot(np.array(train_loss_epoch))
    plt.plot(np.array(val_loss_epoch))
    plt.subplot(1, 3, 2)
    plt.plot(np.array(train_class_loss_epoch))
    plt.plot(np.array(val_class_loss_epoch))
    plt.subplot(1, 3, 3)
    plt.plot(np.array(train_box_loss_epoch))
    plt.plot(np.array(val_box_loss_epoch))
    plt.show()

    print("step: ")
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 3, 1)
    plt.plot(np.array(train_loss_step))
    plt.subplot(1, 3, 2)
    plt.plot(np.array(train_class_loss_step))
    plt.subplot(1, 3, 3)
    plt.plot(np.array(train_box_loss_step))
    plt.show()

    print("epoch duration:", time.time()-epoch_start)


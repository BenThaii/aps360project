from wheat_helpers import *

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
    val_loss, val_class_loss, val_box_loss = check_validation(EffDet, val_loader, anchor_labeler, loss_fn, use_Cuda)
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
    fig = plt.figure(figsize=(20, 4))
    plt.subplot(1, 3, 1)
    plt.plot(np.array(train_loss_epoch))
    plt.plot(np.array(val_loss_epoch))
    plt.subplot(1, 3, 2)
    plt.plot(np.array(train_class_loss_epoch))
    plt.plot(np.array(val_class_loss_epoch))
    plt.subplot(1, 3, 3)
    plt.plot(np.array(train_box_loss_epoch))
    plt.plot(np.array(val_box_loss_epoch))
    plt.close(fig)
    plt.savefig(os.path.join(plot_dir, "epoch{}_epochloss.png".format(epoch)))

    print("step: ")
    fig = plt.figure(figsize=(20, 4))
    plt.subplot(1, 3, 1)
    plt.plot(np.array(train_loss_step))
    plt.subplot(1, 3, 2)
    plt.plot(np.array(train_class_loss_step))
    plt.subplot(1, 3, 3)
    plt.plot(np.array(train_box_loss_step))
    plt.close(fig)
    plt.savefig(os.path.join(plot_dir, "epoch{}_steploss.png".format(epoch)))

    print("epoch duration:", time.time()-epoch_start)


from tqdm import tqdm
from utils.tools import *
from utils.dataLoader import MyDataSet, dataset_collate
from torch.utils.data import DataLoader
from model.anchor_generate import generate_anchors
from model.anchor_match import multibox_target
from model.net import TinySSD
from model.loss import *

# ---------------------------------------------------------
# configuration information
# ---------------------------------------------------------
voc_classes_path = 'C:\\Users\\Omar\\Desktop\\TinySSD_Banana\\model_data\\voc_classes.txt'
image_size_path = 'C:\\Users\\Omar\\Desktop\\TinySSD_Banana\\model_data\\image_size.txt'
train_file_path = '2077_train.txt'
val_file_path = '2077_val.txt'
anchor_sizes_path = 'C:\\Users\\Omar\\Desktop\\TinySSD_Banana\\model_data\\anchor_sizes.txt'
anchor_ratios_path = 'C:\\Users\\Omar\\Desktop\\TinySSD_Banana\\model_data\\anchor_ratios.txt'


def train():
    # ---------------------------------------------------------
    #                   Load training Data
    # ---------------------------------------------------------
    _, num_classes = get_classes(voc_classes_path)
    r = get_image_size(image_size_path)
    with open(train_file_path) as f:
        train_lines = f.readlines()
    train_dataset = MyDataSet(train_lines, r, mode='train')
    train_iter = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, drop_last=True,
                            collate_fn=dataset_collate)

    # ---------------------------------------------------------
    #                   Load validation Data
    # ---------------------------------------------------------
    with open(train_file_path) as f:
        val_lines = f.readlines()
    val_dataset = MyDataSet(val_lines, r, mode='validate')
    val_iter = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True, drop_last=True,
                          collate_fn=dataset_collate)
    # --------------------------- ------------------------------
    #               Generate a prior anchor box
    # ---------------------------------------------------------
    sizes = get_anchor_info(anchor_sizes_path)
    ratios = get_anchor_info(anchor_ratios_path)
    if len(sizes) != len(ratios):
        ratios = [ratios[0]] * len(sizes)
    anchors_per_pixel = len(sizes[0]) + len(ratios[0]) - 1
    feature_map = [r // 8, r // 16, r // 32, r // 64, 1]
    anchors = generate_anchors(feature_map, sizes, ratios)  # (1600+400+100+25+1)*4 anchor boxes

    # ---------------------------------------------------------
    #                       Network Part
    # ---------------------------------------------------------
    net = TinySSD(app=anchors_per_pixel, cn=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # noinspection PyBroadException
    try:
        net.load_state_dict(torch.load('C:\\Users\\Omar\\Desktop\\TinySSD_Banana\\model_data\\result.pt'))
        print("Fine-Tuning...")
    except:
        print("Training from scratch...")
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    validator = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    # trainer = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=5e-5)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, 100)

    # ---------------------------------------------------------
    #                       Start training
    # ---------------------------------------------------------
    num_epochs, timer = 300, Timer()
    timer.start()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net = net.to(device)
    anchors = anchors.to(device)
    cls_loss, bbox_loss = None, None
    for epoch in range(num_epochs):
        print(f' learning rate: {scheduler_lr.get_last_lr()}')
        metric = Accumulator(4)
        net.train()
        for features, target in tqdm(train_iter):
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)

            # Predict the class and offset for each anchor box (multi-scale results are merged)
            cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
            # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

            # Calculate loss function based on predicted and labeled values of class and offset
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), num_classes, bbox_eval(bbox_preds, bbox_labels, bbox_masks), 1)

        for features, target in tqdm(val_iter):
            X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)

            # Predict the class and offset for each anchor box (multi-scale results are merged)
            cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
            # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

            # Calculate loss function based on predicted and labeled values of class and offset
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            validator.step()


        # learning rate decay
        scheduler_lr.step()

        # reserved for display
        cls_loss, bbox_loss = metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_loss, bbox_loss))
        print(f'epoch {epoch + 1}/{num_epochs}: ', 'cls-loss: ', cls_loss, ' box-loss', bbox_loss)

        # Save the trained model for each epoch
        torch.save(net.state_dict(), f'model_data/result_{epoch + 1}.pt')

    print(f'class loss {cls_loss:.2e}, bbox loss {bbox_loss:.2e}')
    print(f'total time: {timer.stop():.1f}s', f' on {str(device)}')


if __name__ == '__main__':
    train()

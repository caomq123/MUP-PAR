# from jrdb import *
#
#
# def return_dataset(cfg):
#
#     train_anns = jrdb_read_dataset_new(cfg.annotation_path, cfg.train_seqs, cfg.num_actions, cfg.num_activities,
#                                        cfg.num_social_activities, is_train=True)   # 读train注释
#     train_frames = jrdb_all_frames(train_anns)    # 读帧(0,15)(0,30)   15帧是固定好的
#
#     test_anns = jrdb_read_dataset_new(cfg.annotation_path, cfg.test_seqs, cfg.num_actions, cfg.num_activities,
#                                       cfg.num_social_activities, is_train=False)   # 读test注释
#     test_frames = jrdb_all_frames(test_anns)
#
#     training_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
#                                 cfg.num_social_activities, train_anns, train_frames,
#                                 cfg.data_path, cfg.image_size, cfg.out_size, num_boxes=cfg.num_boxes,
#                                 num_frames=cfg.num_frames, is_training=True,
#                                 is_finetune=(cfg.training_stage == 1))
#
#     validation_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
#                                   cfg.num_social_activities, test_anns, test_frames,
#                                   cfg.data_path, cfg.image_size, cfg.out_size, num_boxes=cfg.num_boxes,
#                                   num_frames=cfg.num_frames, is_training=False,
#                                   is_finetune=(cfg.training_stage == 1))
#
#
#
#     print('Reading dataset finished...')
#     print('%d train samples' % len(train_frames))
#     print('%d test samples' % len(test_frames))
#
#     return training_set, validation_set

from jrdb import *


def return_dataset(cfg):

    train_anns = jrdb_read_dataset_new(cfg.annotation_path, cfg.train_seqs, cfg.num_actions, cfg.num_activities,
                                       cfg.num_social_activities, is_train=True)   # 读train注释
    train_frames = jrdb_all_frames(train_anns)    # 读帧(0,15)(0,30)   15帧是固定好的

    test_anns = jrdb_read_dataset_new(cfg.annotation_path, cfg.test_seqs, cfg.num_actions, cfg.num_activities,
                                      cfg.num_social_activities, is_train=False)   # 读test注释
    test_frames = jrdb_all_frames(test_anns)

    training_set1 = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                cfg.num_social_activities, train_anns, train_frames,
                                cfg.data_path, cfg.image_size, cfg.out_size, num_boxes=cfg.num_boxes,
                                num_frames=cfg.num_frames, is_training=True,
                                is_finetune=(cfg.training_stage == 1), flag=True)

    training_set2 = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                 cfg.num_social_activities, train_anns, train_frames,
                                 cfg.data_path, cfg.image_size, cfg.out_size, num_boxes=cfg.num_boxes,
                                 num_frames=cfg.num_frames, is_training=True,
                                 is_finetune=(cfg.training_stage == 1), flag=False)       # flag标志图像增强

    validation_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities,
                                  cfg.num_social_activities, test_anns, test_frames,
                                  cfg.data_path, cfg.image_size, cfg.out_size, num_boxes=cfg.num_boxes,
                                  num_frames=cfg.num_frames, is_training=False,
                                  is_finetune=(cfg.training_stage == 1), flag=True)



    print('Reading dataset finished...')
    print('%d train samples' % len(train_frames))
    print('%d test samples' % len(test_frames))

    return training_set1, training_set2, validation_set
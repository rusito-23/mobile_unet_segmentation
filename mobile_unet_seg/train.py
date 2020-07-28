import os
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard
)
from config import _C as cfg
from model import MobileUNet 
from dataset import create_loaders
from log_callback import LoggerCallback
from iou import iou
from log_config import config_logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(cfg):

    # create paths
    tb_logs_path = os.path.join(cfg.OUTPUT_DIR, 'logs')
    chck_path = os.path.join(cfg.OUTPUT_DIR, 'model.h5')
    history_path = os.path.join(cfg.OUTPUT_DIR, 'history.csv')
    logs_path = os.path.join(cfg.OUTPUT_DIR, 'output.log')
    os.mkdir(cfg.OUTPUT_DIR)
    os.mkdir(tb_logs_path)

    # config logger
    config_logger(logs_path)
    log = logging.getLogger('mobile_unet_seg')

    # check cuda
    log.info(f'CUDA AVAILABLE: {tf.test.is_gpu_available()}')

    # create model
    model = MobileUNet(n_classes=cfg.N_CLASSES,
                       in_size=cfg.IN_SIZE,
                       out_size=cfg.OUT_SIZE,
                       weights_url=cfg.WEIGHTS_URL)

    # compile
    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['accuracy', iou]
    )

    # callbacks
    callbacks = [
        TensorBoard(log_dir=tb_logs_path, write_graph=False),
        LoggerCallback(),
        ModelCheckpoint(chck_path, monitor='val_iou', mode='max',
                        save_best_only=True, save_weights_only=True),
        CSVLogger(history_path)
    ]

    # initialize data generators
    train_gen, val_gen = create_loaders(cfg)
    train_steps = len(train_gen) / cfg.BATCH_SIZE
    val_steps = len(val_gen) / cfg.BATCH_SIZE

    # train
    model.fit(train_gen(),
              steps_per_epoch=train_steps,
              validation_data=val_gen(),
              validation_steps=val_steps,
              epochs=cfg.EPOCHS,
              callbacks=callbacks)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    train(cfg)

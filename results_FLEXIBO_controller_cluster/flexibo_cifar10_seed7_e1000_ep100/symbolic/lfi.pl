t(0.50)::action(inc_dropout,overfitting).
t(0.70)::action(dec_conv_layers,overfitting).
t(0.70)::action(dec_conv_block,overfitting).
t(0.70)::action(dec_fc_layers,overfitting).
t(0.40)::action(dec_neurons,overfitting).
t(0.99)::action(data_augmentation,underfitting).
t(0.50)::action(dec_dropout,underfitting).
t(0.50)::action(decr_lr,underfitting).
t(0.70)::action(inc_neurons,underfitting).
t(0.40)::action(new_fc_layers,underfitting).
t(0.70)::action(inc_conv_layers,underfitting).
t(0.40)::action(new_conv_block,underfitting).
t(0.60)::action(decr_lr,inc_loss).
t(0.99)::action(decr_lr,high_lr).
t(0.80)::action(inc_batch_size,floating_loss).
t(0.30)::action(decr_lr,floating_loss).
t(0.70)::action(add_residual, need_skip).
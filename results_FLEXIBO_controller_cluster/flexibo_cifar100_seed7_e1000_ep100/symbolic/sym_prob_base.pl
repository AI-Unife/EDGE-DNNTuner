0.50::action(reg_l2,overfitting):- problem(overfitting).
0.50::action(inc_dropout,overfitting):- problem(overfitting).
0.70::action(dec_conv_layers,overfitting):- problem(overfitting).
0.70::action(dec_conv_block,overfitting):- problem(overfitting).
0.70::action(dec_fc_layers,overfitting):- problem(overfitting).
0.40::action(dec_neurons,overfitting):- problem(overfitting).
0.50::action(remove_residual,overfitting):-problem(overfitting).
0.99::action(data_augmentation,underfitting):- problem(underfitting).
0.50::action(remove_reg_l2,underfitting):- problem(underfitting).
0.50::action(dec_dropout,underfitting):- problem(underfitting).
0.50::action(decr_lr,underfitting):- problem(underfitting).
0.70::action(inc_neurons,underfitting):- problem(underfitting).
0.40::action(new_fc_layers,underfitting):- problem(underfitting).
0.70::action(inc_conv_layers,underfitting):- problem(underfitting).
0.40::action(new_conv_block,underfitting):- problem(underfitting).
0.60::action(decr_lr,inc_loss):- problem(inc_loss).
0.99::action(decr_lr,high_lr):- problem(high_lr).
0.80::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.30::action(decr_lr,floating_loss):- problem(floating_loss).
0.70::action(add_residual, need_skip) :- problem(need_skip).
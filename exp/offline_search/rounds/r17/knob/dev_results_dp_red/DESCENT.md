# 逐任务旋钮下降

开发集使用B 池每任务起点0–29，参考为同一lane 库500 行的减步pure 档，同模型、套件、任务、起点配对。主容差为5 个点，另给0 和 2 个点表。
首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。
推理开销代理为模型调用数除以执行控制数折算的决策数；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。
缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。

档位数字是标定的目标 miss 比例，表中开销是实测值。

## DP · 长任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 0.4 档 | 0.3 档、0.1 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 81.0%，推理开销代理 0.373；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 25 | 26 | 0.519 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 25 | 26 | 0.334 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 26 | 26 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 29 | 26 | 0.133 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 26 | 26 | 0.077 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.544 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 28 | 29 | 0.441 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.344 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.252 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.127 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.042 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 22 | 24 | 0.550 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 21 | 24 | 0.428 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 20 | 24 | 0.331 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 14 | 24 | 0.233 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 7 | 24 | 0.155 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 6 | 24 | 0.108 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 22 | 15 | 0.512 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 21 | 15 | 0.397 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 18 | 15 | 0.303 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 16 | 15 | 0.202 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 15 | 15 | 0.099 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 5 | 15 | 0.000 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 22 | 0.549 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.448 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 22 | 0.337 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 25 | 22 | 0.246 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 22 | 22 | 0.136 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 19 | 22 | 0.070 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 24 | 25 | 0.514 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 20 | 25 | 0.398 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 21 | 25 | 0.302 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 21 | 25 | 0.202 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 11 | 25 | 0.100 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 2 | 25 | 0.000 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 27 | 28 | 0.537 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.418 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 28 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 27 | 28 | 0.122 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 27 | 28 | 0.036 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 26 | 26 | 0.518 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 23 | 26 | 0.433 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 29 | 26 | 0.337 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 24 | 26 | 0.242 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.138 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 22 | 26 | 0.096 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 18 | 21 | 0.526 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 19 | 21 | 0.422 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 16 | 21 | 0.320 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 15 | 21 | 0.232 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 10 | 21 | 0.167 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 6 | 21 | 0.147 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 29 | 0.547 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.434 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.337 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.239 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 28 | 29 | 0.108 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.017 | 通过 |

## DP · 长任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 87.3%，推理开销代理 0.318；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 27 | 26 | 0.523 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 28 | 26 | 0.426 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 25 | 26 | 0.325 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 26 | 26 | 0.230 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 30 | 26 | 0.111 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 28 | 26 | 0.022 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.532 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.334 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.234 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 27 | 29 | 0.126 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.033 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 25 | 24 | 0.532 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 24 | 24 | 0.429 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 23 | 24 | 0.318 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 25 | 24 | 0.214 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 21 | 24 | 0.125 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 24 | 24 | 0.036 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 19 | 15 | 0.538 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 21 | 15 | 0.440 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 22 | 15 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 19 | 15 | 0.136 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 19 | 15 | 0.070 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 22 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 22 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 26 | 22 | 0.235 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 27 | 22 | 0.119 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 28 | 22 | 0.021 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 21 | 25 | 0.540 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 26 | 25 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 24 | 25 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 24 | 25 | 0.241 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 23 | 25 | 0.141 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 22 | 25 | 0.083 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.524 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 30 | 28 | 0.230 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 28 | 0.114 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 28 | 0.009 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 26 | 0.505 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 25 | 26 | 0.416 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 26 | 0.308 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 27 | 26 | 0.217 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 24 | 26 | 0.134 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 25 | 26 | 0.057 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 17 | 21 | 0.535 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 18 | 21 | 0.428 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 15 | 21 | 0.333 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 16 | 21 | 0.231 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 9 | 21 | 0.135 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 4 | 21 | 0.064 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.540 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.446 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.339 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.236 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.117 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.019 | 通过 |

## DP · 长任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 0.5 档 | 0.4 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 0.3 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 86.0%，推理开销代理 0.340；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 27 | 26 | 0.513 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.423 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 27 | 26 | 0.332 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 26 | 26 | 0.233 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 27 | 26 | 0.127 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 27 | 26 | 0.039 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.539 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.334 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.238 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.123 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.016 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 21 | 24 | 0.533 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 24 | 24 | 0.432 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 22 | 24 | 0.339 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 26 | 24 | 0.246 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 27 | 24 | 0.135 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 28 | 24 | 0.041 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 20 | 15 | 0.532 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 24 | 15 | 0.430 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 26 | 15 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 20 | 15 | 0.144 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 22 | 15 | 0.059 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 22 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 22 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 22 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 22 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 28 | 22 | 0.124 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 26 | 22 | 0.027 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 23 | 25 | 0.529 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 25 | 0.428 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 24 | 25 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 26 | 25 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 21 | 25 | 0.145 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 21 | 25 | 0.076 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.536 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 28 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 26 | 28 | 0.330 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 26 | 28 | 0.237 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 28 | 0.125 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 28 | 0.027 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 26 | 0.513 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 26 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 26 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 25 | 26 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 29 | 26 | 0.124 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 25 | 26 | 0.061 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 21 | 21 | 0.528 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 20 | 21 | 0.418 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 18 | 21 | 0.332 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 18 | 21 | 0.233 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 13 | 21 | 0.151 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 16 | 21 | 0.077 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.534 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.429 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.326 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.223 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.107 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.005 | 通过 |

## DP · 长任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 88.3%，推理开销代理 0.099；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 29 | 26 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 27 | 26 | 0.435 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 29 | 26 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 26 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 26 | 0.127 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 26 | 0.011 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.537 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.426 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.332 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.121 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.019 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 24 | 24 | 0.536 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 26 | 24 | 0.441 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 26 | 24 | 0.342 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 24 | 24 | 0.243 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 27 | 24 | 0.139 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 26 | 24 | 0.047 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 21 | 15 | 0.525 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 18 | 15 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 18 | 15 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 23 | 15 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 23 | 15 | 0.129 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 22 | 15 | 0.042 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 25 | 22 | 0.531 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 22 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 22 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 27 | 22 | 0.124 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 27 | 22 | 0.018 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 25 | 25 | 0.534 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 25 | 25 | 0.424 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 26 | 25 | 0.330 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 19 | 25 | 0.240 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 26 | 25 | 0.131 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 26 | 25 | 0.043 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.533 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.425 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 28 | 28 | 0.235 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 28 | 0.120 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 28 | 28 | 0.028 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 26 | 26 | 0.513 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 28 | 26 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 26 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 26 | 0.232 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.130 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 26 | 26 | 0.049 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 20 | 21 | 0.528 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 22 | 21 | 0.433 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 21 | 21 | 0.330 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 19 | 21 | 0.236 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 17 | 21 | 0.148 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 18 | 21 | 0.059 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.532 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 28 | 29 | 0.442 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.336 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.237 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.118 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.009 | 通过 |

## DP · 长任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 0.5 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 0.3 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、关（只有停滞门） | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 90.0%，推理开销代理 0.265；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 29 | 26 | 0.526 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 25 | 26 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 27 | 26 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 23 | 26 | 0.147 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 29 | 26 | 0.020 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.531 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.434 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 30 | 29 | 0.114 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.013 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 24 | 24 | 0.537 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 26 | 24 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 27 | 24 | 0.342 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 24 | 24 | 0.243 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 23 | 24 | 0.152 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 25 | 24 | 0.066 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 22 | 15 | 0.529 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 19 | 15 | 0.429 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 21 | 15 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.244 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 23 | 15 | 0.138 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 26 | 15 | 0.038 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 22 | 0.533 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 22 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 26 | 22 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 25 | 22 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 29 | 22 | 0.121 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 29 | 22 | 0.012 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 22 | 25 | 0.539 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 21 | 25 | 0.426 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 25 | 25 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 25 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 25 | 25 | 0.141 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 26 | 25 | 0.053 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.517 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.436 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 26 | 28 | 0.328 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 26 | 28 | 0.242 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 28 | 0.119 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 28 | 0.013 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 29 | 26 | 0.504 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 26 | 26 | 0.428 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 26 | 0.326 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 29 | 26 | 0.228 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.128 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 28 | 26 | 0.030 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 22 | 21 | 0.535 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 18 | 21 | 0.426 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 26 | 21 | 0.323 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 22 | 21 | 0.231 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 17 | 21 | 0.141 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 22 | 21 | 0.047 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.532 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 29 | 29 | 0.443 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.336 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.238 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.114 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.007 | 通过 |

## DP · 长任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、关（只有停滞门） | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 90.3%，推理开销代理 0.144；同任务纯推理成功率 81.7%，开销代理 1.000。全十任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 28 | 26 | 0.520 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 27 | 26 | 0.435 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 27 | 26 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 25 | 26 | 0.243 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 26 | 26 | 0.134 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 26 | 0.009 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.550 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.424 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.333 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.234 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 30 | 29 | 0.113 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.018 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 26 | 24 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 26 | 24 | 0.439 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 23 | 24 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 26 | 24 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 26 | 24 | 0.142 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 24 | 24 | 0.060 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 24 | 15 | 0.535 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 23 | 15 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 22 | 15 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 19 | 15 | 0.242 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 25 | 15 | 0.129 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 23 | 15 | 0.046 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 22 | 0.529 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.432 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 22 | 0.325 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 30 | 22 | 0.233 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 29 | 22 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 28 | 22 | 0.014 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 25 | 25 | 0.532 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 23 | 25 | 0.431 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 27 | 25 | 0.330 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 25 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 24 | 25 | 0.136 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 26 | 25 | 0.059 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 28 | 28 | 0.529 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 28 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 28 | 0.127 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 28 | 0.006 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 26 | 0.495 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 26 | 0.411 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 29 | 26 | 0.315 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 26 | 0.221 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 28 | 26 | 0.117 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 29 | 26 | 0.015 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 23 | 21 | 0.524 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 19 | 21 | 0.423 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 26 | 21 | 0.325 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 24 | 21 | 0.222 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 18 | 21 | 0.137 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 21 | 21 | 0.027 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.538 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.439 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.336 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.235 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.110 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.007 | 通过 |

## DP · 空间任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 86.3%，推理开销代理 0.521；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 17 | 20 | 0.497 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.401 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.306 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 12 | 20 | 0.204 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 20 | 0.101 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 9 | 20 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 26 | 29 | 0.547 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 29 | 0.339 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 26 | 29 | 0.239 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 29 | 0.150 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 29 | 0.095 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 25 | 27 | 0.535 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.437 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.251 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.138 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.028 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.572 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.446 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.354 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.244 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.122 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.031 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 19 | 0.541 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 19 | 0.421 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 19 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 18 | 19 | 0.245 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.163 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 19 | 0.115 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 25 | 27 | 0.567 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 25 | 27 | 0.441 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.255 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.135 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 27 | 0.056 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 28 | 0.554 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.452 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 28 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 28 | 0.245 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 28 | 0.184 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 28 | 0.094 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 16 | 0.492 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 16 | 0.401 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 17 | 16 | 0.294 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 16 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 16 | 0.103 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 16 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 28 | 0.566 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 28 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 28 | 0.248 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 27 | 0.452 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.350 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 27 | 0.248 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 27 | 0.132 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 23 | 27 | 0.071 | 失败 |

## DP · 空间任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.0%，推理开销代理 0.153；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.540 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.430 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 20 | 0.324 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.247 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 14 | 20 | 0.180 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 20 | 0.154 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.554 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 29 | 0.448 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 29 | 0.235 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.126 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.025 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.526 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.558 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.348 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.235 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 30 | 0.047 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 19 | 0.523 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 19 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 19 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 19 | 0.142 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 21 | 19 | 0.064 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 27 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 26 | 27 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.107 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 27 | 0.021 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.498 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.314 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 28 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 28 | 0.129 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 23 | 28 | 0.051 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 16 | 0.530 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 16 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 17 | 16 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 24 | 16 | 0.244 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 16 | 0.145 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 16 | 0.063 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.561 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.457 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.244 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.526 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.331 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 27 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 27 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 27 | 0.034 | 通过 |

## DP · 空间任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.7%，推理开销代理 0.081；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.543 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 20 | 0.438 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.255 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.165 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.566 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.448 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.135 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.027 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.525 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 26 | 27 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.581 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.446 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 30 | 0.345 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.109 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.016 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 19 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.153 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.072 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.549 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.443 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.347 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.115 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.021 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.539 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.444 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.332 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.224 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 28 | 0.035 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 16 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 16 | 0.444 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 16 | 0.335 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 26 | 16 | 0.246 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 16 | 0.150 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 16 | 0.075 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.342 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.446 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 27 | 0.320 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.240 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 27 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.018 | 通过 |

## DP · 空间任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 87.0%，推理开销代理 0.259；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 20 | 0.531 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.246 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 20 | 0.168 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.571 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.317 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.229 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.120 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.031 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.525 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 27 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.229 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.126 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.579 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.449 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.342 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.105 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 19 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 19 | 0.443 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 19 | 0.347 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 19 | 0.253 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 19 | 0.143 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.080 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.575 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.446 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 27 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.115 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.005 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 26 | 28 | 0.447 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 28 | 0.330 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 28 | 0.204 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 16 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 16 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 14 | 16 | 0.335 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 16 | 0.250 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 23 | 16 | 0.164 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 23 | 16 | 0.091 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.547 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.433 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.110 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.318 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 27 | 0.038 | 失败 |

## DP · 空间任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 85.0%，推理开销代理 0.134；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 20 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.248 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 18 | 20 | 0.170 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 20 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.564 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 29 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.124 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.032 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.444 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 27 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 27 | 0.023 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.594 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.349 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 30 | 0.228 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.104 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.550 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 19 | 0.437 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 19 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.239 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.153 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.082 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.568 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.453 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.228 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.012 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 28 | 0.324 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.097 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.004 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 16 | 0.516 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 15 | 16 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 14 | 16 | 0.334 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 16 | 0.246 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 16 | 0.167 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 16 | 0.111 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.555 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.433 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.523 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.433 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.319 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.112 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 27 | 0.031 | 通过 |

## DP · 空间任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.3 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 87.3%，推理开销代理 0.205；同任务纯推理成功率 83.7%，开销代理 1.000。全十任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.544 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.346 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.243 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.163 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 21 | 20 | 0.114 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 26 | 29 | 0.565 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 29 | 0.442 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.330 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 29 | 0.239 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.119 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.024 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.527 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 27 | 0.447 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.009 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.583 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 30 | 0.341 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 30 | 0.230 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.130 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.011 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.534 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 19 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 19 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 19 | 0.239 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 19 | 0.155 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.080 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.574 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 27 | 0.447 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 27 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 27 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.003 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.547 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.443 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.320 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.101 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.015 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 16 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 16 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 15 | 16 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 16 | 0.253 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 16 | 0.175 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 16 | 0.124 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.554 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.003 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.220 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.117 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 27 | 0.015 | 通过 |

## 汇总

| 模型 | 任务套件 | 库集数 | 已冻结任务数 | 拼接局数 | 拼接成功率 | 推理开销代理 | 同任务纯推理成功率 |
|---|---|---:|---:|---:|---:|---:|---:|
| DP | 长任务 | 50 | 10 | 300 | 81.0% | 0.373 | 81.7% |
| DP | 长任务 | 100 | 10 | 300 | 87.3% | 0.318 | 81.7% |
| DP | 长任务 | 200 | 10 | 300 | 86.0% | 0.340 | 81.7% |
| DP | 长任务 | 300 | 10 | 300 | 88.3% | 0.099 | 81.7% |
| DP | 长任务 | 400 | 10 | 300 | 90.0% | 0.265 | 81.7% |
| DP | 长任务 | 500 | 10 | 300 | 90.3% | 0.144 | 81.7% |
| DP | 空间任务 | 50 | 10 | 300 | 86.3% | 0.521 | 83.7% |
| DP | 空间任务 | 100 | 10 | 300 | 89.0% | 0.153 | 83.7% |
| DP | 空间任务 | 200 | 10 | 300 | 89.7% | 0.081 | 83.7% |
| DP | 空间任务 | 300 | 10 | 300 | 87.0% | 0.259 | 83.7% |
| DP | 空间任务 | 400 | 10 | 300 | 85.0% | 0.134 | 83.7% |
| DP | 空间任务 | 500 | 10 | 300 | 87.3% | 0.205 | 83.7% |

# 逐任务旋钮下降

开发集使用B 池每任务起点0–29，参考为同一lane 库500 行的减步pure 档，同模型、套件、任务、起点配对。主容差为5 个点，另给0 和 2 个点表。
首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。
推理开销代理为模型调用数除以执行控制数折算的决策数；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。
缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。

档位数字是标定的目标 miss 比例，表中开销是实测值。

## Cosmos · 长任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 0.4 档 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 97.0%，推理开销代理 0.862；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 28 | 30 | 0.539 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 28 | 30 | 0.434 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 24 | 30 | 0.342 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 22 | 30 | 0.276 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 16 | 30 | 0.241 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 24 | 30 | 0.102 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 27 | 29 | 0.535 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 24 | 29 | 0.454 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 17 | 29 | 0.381 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 20 | 29 | 0.295 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 21 | 29 | 0.214 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 21 | 29 | 0.174 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 22 | 28 | 0.573 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 21 | 28 | 0.479 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 20 | 28 | 0.392 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 13 | 28 | 0.316 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 19 | 28 | 0.242 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 18 | 28 | 0.220 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 28 | 25 | 0.539 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 21 | 25 | 0.458 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 21 | 25 | 0.356 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 19 | 25 | 0.284 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 16 | 25 | 0.208 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 16 | 25 | 0.181 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 24 | 30 | 0.527 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 15 | 30 | 0.461 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 11 | 30 | 0.380 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 5 | 30 | 0.341 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 2 | 30 | 0.270 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 1 | 30 | 0.269 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 29 | 29 | 0.539 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 25 | 29 | 0.444 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 23 | 29 | 0.353 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 13 | 29 | 0.290 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 8 | 29 | 0.245 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 5 | 29 | 0.250 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 28 | 30 | 0.540 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 30 | 0.434 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 20 | 30 | 0.366 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 20 | 30 | 0.270 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 8 | 30 | 0.251 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 7 | 30 | 0.227 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 21 | 29 | 0.544 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 14 | 29 | 0.476 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 8 | 29 | 0.410 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 6 | 29 | 0.366 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 4 | 29 | 0.338 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 3 | 29 | 0.331 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 24 | 29 | 0.537 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 20 | 29 | 0.454 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 10 | 29 | 0.383 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 7 | 29 | 0.308 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 3 | 29 | 0.254 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 4 | 29 | 0.220 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 30 | 0.534 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 28 | 30 | 0.436 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 26 | 30 | 0.343 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 14 | 30 | 0.361 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 10 | 30 | 0.347 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 10 | 30 | 0.286 | 失败 |

## Cosmos · 长任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 0.4 档 | 0.1 档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 0.5 档 | 0.1 档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 0.4 档 | 0.3 档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 0.2 档 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 96.7%，推理开销代理 0.598；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.521 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 28 | 30 | 0.427 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 26 | 30 | 0.343 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 26 | 30 | 0.247 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 30 | 30 | 0.100 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 28 | 30 | 0.052 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 27 | 29 | 0.530 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 26 | 29 | 0.425 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 27 | 29 | 0.344 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 24 | 29 | 0.288 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.135 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 26 | 29 | 0.116 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 28 | 28 | 0.523 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 25 | 28 | 0.416 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 27 | 28 | 0.318 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 26 | 28 | 0.215 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 21 | 28 | 0.143 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 24 | 28 | 0.053 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 29 | 25 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 28 | 25 | 0.415 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 28 | 25 | 0.327 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 22 | 25 | 0.237 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 22 | 25 | 0.147 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 20 | 25 | 0.077 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 25 | 30 | 0.537 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 26 | 30 | 0.436 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 26 | 30 | 0.340 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 15 | 30 | 0.325 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 14 | 30 | 0.265 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 13 | 30 | 0.234 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 28 | 29 | 0.535 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 29 | 0.428 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 24 | 29 | 0.335 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 21 | 29 | 0.250 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 11 | 29 | 0.159 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 17 | 29 | 0.082 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.527 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.422 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.312 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 27 | 30 | 0.207 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 30 | 0.133 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 25 | 30 | 0.017 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 29 | 0.526 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 25 | 29 | 0.440 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 19 | 29 | 0.384 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 14 | 29 | 0.327 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 9 | 29 | 0.290 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 9 | 29 | 0.263 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 27 | 29 | 0.535 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 24 | 29 | 0.443 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 17 | 29 | 0.377 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 16 | 29 | 0.325 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 13 | 29 | 0.267 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 8 | 29 | 0.250 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 30 | 0.520 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 29 | 30 | 0.432 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 30 | 0.294 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 26 | 30 | 0.269 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 25 | 30 | 0.224 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 25 | 30 | 0.142 | 失败 |

## Cosmos · 长任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 0.5 档 | 0.3 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 0.3 档 | 0.2 档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 95.3%，推理开销代理 0.475；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.518 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 30 | 0.426 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 28 | 30 | 0.337 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 28 | 30 | 0.229 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 26 | 30 | 0.163 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 27 | 30 | 0.070 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 27 | 29 | 0.531 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 25 | 29 | 0.434 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.329 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 25 | 29 | 0.283 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 25 | 29 | 0.177 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 28 | 29 | 0.065 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 29 | 28 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 28 | 28 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 24 | 28 | 0.336 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 27 | 28 | 0.218 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 26 | 28 | 0.145 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 25 | 28 | 0.047 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 29 | 25 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 28 | 25 | 0.323 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 27 | 25 | 0.232 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 27 | 25 | 0.138 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 21 | 25 | 0.057 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 28 | 30 | 0.529 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 30 | 0.434 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 30 | 0.337 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 26 | 30 | 0.250 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 24 | 30 | 0.171 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 25 | 30 | 0.122 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 28 | 29 | 0.534 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 29 | 29 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 28 | 29 | 0.325 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 22 | 29 | 0.262 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 19 | 29 | 0.169 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 20 | 29 | 0.109 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.527 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 30 | 0.315 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.207 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.129 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 30 | 0.018 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 29 | 0.520 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 29 | 0.421 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 28 | 29 | 0.322 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 29 | 0.211 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 29 | 0.129 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 26 | 29 | 0.077 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 27 | 29 | 0.530 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 28 | 29 | 0.446 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 20 | 29 | 0.376 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 14 | 29 | 0.351 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 18 | 29 | 0.276 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 14 | 29 | 0.249 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 30 | 0.499 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 30 | 0.425 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 30 | 0.293 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 29 | 30 | 0.204 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 29 | 30 | 0.120 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 30 | 0.029 | 通过 |

## Cosmos · 长任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 0.4 档 | 0.2 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 0.3 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 0.4 档 | 0.3 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 96.0%，推理开销代理 0.305；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.503 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 28 | 30 | 0.431 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 26 | 30 | 0.345 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.208 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 27 | 30 | 0.133 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.007 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 30 | 29 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 28 | 29 | 0.418 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.334 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 28 | 29 | 0.251 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.140 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.022 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 27 | 28 | 0.526 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 29 | 28 | 0.421 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 28 | 28 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 28 | 28 | 0.229 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 26 | 28 | 0.160 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 26 | 28 | 0.068 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 28 | 25 | 0.521 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 30 | 25 | 0.426 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 28 | 25 | 0.327 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 28 | 25 | 0.231 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 27 | 25 | 0.141 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 26 | 25 | 0.048 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 30 | 0.512 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 26 | 30 | 0.430 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 30 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 25 | 30 | 0.162 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 24 | 30 | 0.104 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 29 | 29 | 0.526 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 29 | 0.448 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 29 | 29 | 0.338 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 22 | 29 | 0.262 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 26 | 29 | 0.147 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 24 | 29 | 0.080 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.518 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.317 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.206 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.113 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 30 | 0.004 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 30 | 29 | 0.514 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 29 | 29 | 0.417 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 30 | 29 | 0.314 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 28 | 29 | 0.225 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 26 | 29 | 0.144 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 24 | 29 | 0.110 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 28 | 29 | 0.525 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 27 | 29 | 0.438 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 22 | 29 | 0.373 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 18 | 29 | 0.325 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 20 | 29 | 0.272 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 14 | 29 | 0.235 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 30 | 0.510 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 30 | 0.412 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 30 | 0.286 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 30 | 0.182 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 30 | 0.091 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |

## Cosmos · 长任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 0.3 档 | 0.2 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 0.3 档 | 0.2 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 0.3 档 | 0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 96.3%，推理开销代理 0.419；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.512 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 30 | 0.416 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 27 | 30 | 0.331 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 30 | 0.220 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 30 | 0.106 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.522 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.414 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 25 | 29 | 0.356 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.230 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 25 | 29 | 0.174 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.031 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 25 | 28 | 0.521 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 29 | 28 | 0.410 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 28 | 28 | 0.328 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 29 | 28 | 0.219 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 25 | 28 | 0.173 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 22 | 28 | 0.111 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 30 | 25 | 0.520 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 26 | 25 | 0.426 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 24 | 25 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 30 | 25 | 0.217 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 28 | 25 | 0.130 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 26 | 25 | 0.045 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 29 | 30 | 0.521 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 26 | 30 | 0.418 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 30 | 0.330 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 25 | 30 | 0.248 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 22 | 30 | 0.173 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 21 | 30 | 0.112 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 28 | 29 | 0.528 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 28 | 29 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 30 | 29 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 25 | 29 | 0.255 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 24 | 29 | 0.158 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 23 | 29 | 0.084 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.518 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.420 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.311 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 30 | 30 | 0.205 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 30 | 30 | 0.115 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 30 | 0.002 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 29 | 29 | 0.516 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 29 | 29 | 0.422 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 29 | 0.342 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 28 | 29 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 23 | 29 | 0.177 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 23 | 29 | 0.121 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 25 | 29 | 0.530 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 28 | 29 | 0.435 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 21 | 29 | 0.370 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 16 | 29 | 0.336 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 17 | 29 | 0.275 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 18 | 29 | 0.210 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 30 | 0.514 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 30 | 0.410 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 30 | 0.293 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 30 | 0.181 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 30 | 0.090 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |

## Cosmos · 长任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 0.3 档 | 0.2 档、0.1 档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 95.7%，推理开销代理 0.359；同任务纯推理成功率 96.3%，开销代理 1.000。全十任务纯推理成功率 96.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.511 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 30 | 0.422 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 28 | 30 | 0.327 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.206 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 29 | 30 | 0.098 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 28 | 30 | 0.042 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 30 | 29 | 0.526 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.408 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.329 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 27 | 29 | 0.146 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 28 | 29 | 0.052 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 29 | 28 | 0.525 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 30 | 28 | 0.416 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 27 | 28 | 0.342 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 28 | 28 | 0.226 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 28 | 28 | 0.148 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 28 | 28 | 0.044 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 29 | 25 | 0.517 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.430 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 27 | 25 | 0.326 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 28 | 25 | 0.224 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 26 | 25 | 0.142 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 23 | 25 | 0.052 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 30 | 0.524 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 30 | 0.427 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 30 | 0.324 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 30 | 0.233 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 25 | 30 | 0.171 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 25 | 30 | 0.116 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 29 | 29 | 0.523 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 28 | 29 | 0.432 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 27 | 29 | 0.330 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 28 | 29 | 0.238 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 27 | 29 | 0.162 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 26 | 29 | 0.085 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.519 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.422 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.310 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.206 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 30 | 0.008 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 29 | 0.520 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 29 | 29 | 0.419 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 29 | 29 | 0.315 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 27 | 29 | 0.248 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 26 | 29 | 0.137 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 24 | 29 | 0.103 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 27 | 29 | 0.521 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 27 | 29 | 0.434 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 23 | 29 | 0.365 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 18 | 29 | 0.329 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 29 | 0.265 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 15 | 29 | 0.246 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 30 | 0.515 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 30 | 0.412 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 30 | 0.293 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 30 | 0.179 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 30 | 0.090 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 30 | 0.003 | 通过 |

## Cosmos · 空间任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.1 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 88.3%，推理开销代理 0.697；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 16 | 19 | 0.549 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 11 | 19 | 0.407 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 8 | 19 | 0.325 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 4 | 19 | 0.220 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 1 | 19 | 0.145 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 0 | 19 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 29 | 0.541 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.359 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 23 | 29 | 0.312 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 29 | 0.260 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 29 | 0.243 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 26 | 30 | 0.576 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 25 | 30 | 0.464 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 25 | 30 | 0.337 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 12 | 30 | 0.325 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 0 | 30 | 0.276 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 0 | 30 | 0.245 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.665 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.503 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.319 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.208 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.208 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 21 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 21 | 0.417 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 21 | 0.362 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 18 | 21 | 0.299 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 18 | 21 | 0.157 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 21 | 0.120 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.550 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.430 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.337 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 18 | 30 | 0.329 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 10 | 30 | 0.272 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 7 | 30 | 0.236 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.599 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.447 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 26 | 29 | 0.397 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 29 | 0.340 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 25 | 29 | 0.168 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 24 | 29 | 0.175 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 22 | 22 | 0.580 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 17 | 22 | 0.441 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 10 | 22 | 0.428 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 7 | 22 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 6 | 22 | 0.283 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 6 | 22 | 0.283 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.559 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 26 | 30 | 0.445 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.335 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 30 | 0.314 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.152 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 30 | 0.036 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 15 | 30 | 0.583 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 12 | 30 | 0.457 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 9 | 30 | 0.398 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 6 | 30 | 0.359 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 3 | 30 | 0.316 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 3 | 30 | 0.297 | 失败 |

## Cosmos · 空间任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.3 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 91.3%，推理开销代理 0.492；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 17 | 19 | 0.565 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 16 | 19 | 0.474 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 19 | 0.375 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 19 | 0.299 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 19 | 0.213 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 19 | 0.184 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 29 | 0.534 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 29 | 0.408 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.349 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 29 | 0.023 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.550 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 30 | 0.440 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.334 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.254 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.008 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.230 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.163 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 30 | 0.061 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 21 | 0.547 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 21 | 0.428 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 21 | 0.372 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 18 | 21 | 0.282 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 18 | 21 | 0.178 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 21 | 0.155 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.515 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.405 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.324 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.178 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.172 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.574 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.431 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.317 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 22 | 29 | 0.280 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 29 | 0.079 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 29 | 0.073 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 22 | 0.555 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 22 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 22 | 0.367 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 22 | 0.246 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 22 | 0.051 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 22 | 0.034 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.559 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.428 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.302 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.249 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.144 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.565 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.317 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.273 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 30 | 0.156 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 30 | 0.054 | 失败 |

## Cosmos · 空间任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.0%，推理开销代理 0.243；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 19 | 0.533 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.449 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 19 | 0.368 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 14 | 19 | 0.302 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 16 | 19 | 0.202 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 19 | 0.191 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.431 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.365 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.214 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.065 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.037 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.535 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.320 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.259 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.128 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.004 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 30 | 0.536 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.317 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.197 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.164 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.005 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 21 | 0.550 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 21 | 0.450 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 21 | 0.381 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 21 | 0.287 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 21 | 0.217 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 21 | 0.159 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.507 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.411 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.300 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.174 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.156 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.005 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.575 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.427 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.342 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.049 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.025 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 25 | 22 | 0.538 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 23 | 22 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 22 | 0.412 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 22 | 0.293 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 22 | 22 | 0.199 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 22 | 22 | 0.182 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.543 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.326 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.255 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.151 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 30 | 0.044 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.561 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.319 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.268 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.134 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |

## Cosmos · 空间任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 88.3%，推理开销代理 0.148；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 19 | 0.534 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.466 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 19 | 0.373 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 19 | 0.309 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 18 | 19 | 0.228 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 19 | 0.219 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.525 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 29 | 0.435 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 26 | 29 | 0.376 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.198 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.071 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.554 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.307 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.264 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.131 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.004 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.519 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.329 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.202 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.174 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.025 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 21 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 21 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 21 | 0.391 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 21 | 0.288 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 21 | 0.210 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 21 | 0.148 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.494 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.417 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.299 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.167 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.170 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.578 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.329 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.086 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.036 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 22 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 22 | 0.455 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 22 | 0.407 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 22 | 0.326 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 22 | 0.209 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 22 | 0.197 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.561 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.450 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.322 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.263 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.149 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.556 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.301 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.268 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.144 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.013 | 通过 |

## Cosmos · 空间任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.7%，推理开销代理 0.251；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.547 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.468 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 19 | 0.378 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 19 | 0.318 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 19 | 0.233 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 19 | 0.220 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.428 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.372 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 29 | 0.214 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.074 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.531 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.322 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.262 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.135 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.004 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.526 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.342 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.179 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.167 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 21 | 0.548 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 21 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 21 | 0.384 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 21 | 0.277 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 21 | 0.194 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 21 | 21 | 0.155 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.503 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.412 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.299 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.169 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.152 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.583 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 29 | 0.349 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 29 | 0.202 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 29 | 0.052 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 29 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 22 | 0.563 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 22 | 0.457 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 22 | 0.415 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 22 | 0.328 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 22 | 0.247 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 22 | 0.232 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.443 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.317 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.257 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.141 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.420 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.302 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.263 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.135 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |

## Cosmos · 空间任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.7%，推理开销代理 0.131；同任务纯推理成功率 90.0%，开销代理 1.000。全十任务纯推理成功率 90.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.454 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 19 | 0.372 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 19 | 0.324 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 15 | 19 | 0.226 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 19 | 0.215 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.528 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.433 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.372 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 29 | 0.186 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 29 | 0.039 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.309 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.260 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.132 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.515 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.323 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.188 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.172 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.039 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 21 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 21 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 21 | 0.391 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 21 | 0.288 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 21 | 0.186 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 21 | 21 | 0.149 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.517 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.413 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.305 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.170 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.162 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.562 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.422 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 29 | 0.358 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.214 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 29 | 0.032 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 29 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 22 | 22 | 0.578 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 17 | 22 | 0.479 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 22 | 0.424 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 22 | 0.326 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 22 | 0.257 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 22 | 0.242 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.452 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.307 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.264 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.142 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.558 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.420 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.310 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.265 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.134 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |

## 汇总

| 模型 | 任务套件 | 库集数 | 已冻结任务数 | 拼接局数 | 拼接成功率 | 推理开销代理 | 同任务纯推理成功率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cosmos | 长任务 | 50 | 10 | 300 | 97.0% | 0.862 | 96.3% |
| Cosmos | 长任务 | 100 | 10 | 300 | 96.7% | 0.598 | 96.3% |
| Cosmos | 长任务 | 200 | 10 | 300 | 95.3% | 0.475 | 96.3% |
| Cosmos | 长任务 | 300 | 10 | 300 | 96.0% | 0.305 | 96.3% |
| Cosmos | 长任务 | 400 | 10 | 300 | 96.3% | 0.419 | 96.3% |
| Cosmos | 长任务 | 500 | 10 | 300 | 95.7% | 0.359 | 96.3% |
| Cosmos | 空间任务 | 50 | 10 | 300 | 88.3% | 0.697 | 90.0% |
| Cosmos | 空间任务 | 100 | 10 | 300 | 91.3% | 0.492 | 90.0% |
| Cosmos | 空间任务 | 200 | 10 | 300 | 89.0% | 0.243 | 90.0% |
| Cosmos | 空间任务 | 300 | 10 | 300 | 88.3% | 0.148 | 90.0% |
| Cosmos | 空间任务 | 400 | 10 | 300 | 89.7% | 0.251 | 90.0% |
| Cosmos | 空间任务 | 500 | 10 | 300 | 89.7% | 0.131 | 90.0% |

# 逐任务旋钮下降

开发集使用B 池每任务起点0–29，纯推理参考来自教师库。主容差为5 个点，另给0 和 2 个点表。
首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。
推理开销代理为模型调用数除以执行控制数折算的决策数；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。
缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。

档位数字是标定的目标 miss 比例，表中开销是实测值。

## Fast-WAM · 长任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 0.3 档 | 0.1 档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 93.3%，推理开销代理 0.383；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.532 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 30 | 0.443 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 29 | 30 | 0.341 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 30 | 0.246 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 30 | 0.136 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.029 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.414 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 27 | 29 | 0.308 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 26 | 29 | 0.209 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.103 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 25 | 29 | 0.043 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 29 | 28 | 0.562 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 30 | 28 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 30 | 28 | 0.342 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 28 | 28 | 0.247 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 25 | 28 | 0.145 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 27 | 28 | 0.070 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 29 | 25 | 0.535 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.413 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 27 | 25 | 0.315 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 25 | 0.223 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 18 | 25 | 0.128 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 14 | 25 | 0.029 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 28 | 0.537 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 29 | 28 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 26 | 28 | 0.329 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 26 | 28 | 0.236 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 23 | 28 | 0.140 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 13 | 28 | 0.115 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 26 | 27 | 0.539 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 27 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 26 | 27 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 27 | 0.245 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 18 | 27 | 0.151 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 13 | 27 | 0.109 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.533 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 30 | 0.444 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.339 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 28 | 30 | 0.246 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 30 | 0.130 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 26 | 30 | 0.045 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 27 | 28 | 0.525 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 29 | 28 | 0.447 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 25 | 28 | 0.339 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 22 | 28 | 0.240 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 14 | 28 | 0.180 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 13 | 28 | 0.151 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 28 | 28 | 0.552 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 26 | 28 | 0.435 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 25 | 28 | 0.341 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 24 | 28 | 0.245 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 11 | 28 | 0.188 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 14 | 28 | 0.140 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.560 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.454 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.336 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.238 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 29 | 29 | 0.141 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 28 | 29 | 0.078 | 通过 |

## Fast-WAM · 长任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 0.4 档 | 0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.3 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 94.7%，推理开销代理 0.371；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.541 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 30 | 30 | 0.448 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 30 | 30 | 0.333 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.233 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 29 | 30 | 0.123 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.016 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.515 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.412 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 30 | 29 | 0.301 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.208 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.102 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.019 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 28 | 28 | 0.556 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 29 | 28 | 0.439 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 30 | 28 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 29 | 28 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 28 | 28 | 0.135 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 28 | 28 | 0.060 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 30 | 25 | 0.537 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.432 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 29 | 25 | 0.342 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 25 | 25 | 0.238 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 19 | 25 | 0.135 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 17 | 25 | 0.039 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 28 | 0.531 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.422 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 28 | 0.317 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.221 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 17 | 28 | 0.138 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 9 | 28 | 0.087 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 28 | 27 | 0.545 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 27 | 0.428 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 24 | 27 | 0.337 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 24 | 27 | 0.240 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 16 | 27 | 0.157 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 12 | 27 | 0.117 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 30 | 0.435 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 30 | 30 | 0.234 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 30 | 0.125 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 24 | 30 | 0.024 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 29 | 28 | 0.528 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 25 | 28 | 0.434 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 28 | 0.331 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 27 | 28 | 0.240 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 20 | 28 | 0.159 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 18 | 28 | 0.105 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 26 | 28 | 0.546 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 24 | 28 | 0.427 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 27 | 28 | 0.341 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 20 | 28 | 0.242 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 28 | 0.156 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 16 | 28 | 0.105 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 29 | 0.552 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.443 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.325 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.242 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.125 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.035 | 通过 |

## Fast-WAM · 长任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 0.4 档 | 0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 93.0%，推理开销代理 0.233；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.517 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 30 | 30 | 0.429 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 30 | 30 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.235 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 30 | 0.131 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 29 | 30 | 0.018 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.551 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 28 | 29 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.231 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.124 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 27 | 29 | 0.033 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 30 | 28 | 0.541 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 28 | 28 | 0.434 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 30 | 28 | 0.341 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 28 | 28 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 29 | 28 | 0.134 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 28 | 28 | 0.037 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 30 | 25 | 0.531 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 27 | 25 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 27 | 25 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 24 | 25 | 0.126 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 20 | 25 | 0.040 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.523 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 29 | 28 | 0.240 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 24 | 28 | 0.134 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 14 | 28 | 0.087 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 26 | 27 | 0.533 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 27 | 0.422 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 26 | 27 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 26 | 27 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 21 | 27 | 0.142 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 12 | 27 | 0.087 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.536 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 30 | 0.440 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 30 | 0.335 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.235 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 30 | 0.126 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 24 | 30 | 0.040 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 28 | 0.519 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 26 | 28 | 0.432 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 28 | 0.328 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 28 | 28 | 0.229 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 19 | 28 | 0.159 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 17 | 28 | 0.101 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 27 | 28 | 0.536 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 26 | 28 | 0.434 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 26 | 28 | 0.340 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 20 | 28 | 0.241 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 28 | 0.160 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 16 | 28 | 0.103 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.551 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.443 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.341 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.238 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.120 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.039 | 通过 |

## Fast-WAM · 长任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 0.5 档 | 0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 0.4 档 | 0.3 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 93.3%，推理开销代理 0.269；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.526 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 30 | 30 | 0.429 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 30 | 30 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 29 | 30 | 0.121 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 28 | 30 | 0.021 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.544 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 28 | 29 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 28 | 29 | 0.329 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.226 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 27 | 29 | 0.125 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.005 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 29 | 28 | 0.531 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 30 | 28 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 28 | 28 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 30 | 28 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 26 | 28 | 0.136 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 25 | 28 | 0.063 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 28 | 25 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.436 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 27 | 25 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 26 | 25 | 0.238 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 25 | 25 | 0.131 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 24 | 25 | 0.042 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 28 | 0.525 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 28 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.235 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 20 | 28 | 0.151 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 11 | 28 | 0.105 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 26 | 27 | 0.535 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 26 | 27 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 28 | 27 | 0.336 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 27 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 22 | 27 | 0.143 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 18 | 27 | 0.061 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.523 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 30 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.335 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 30 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.122 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 27 | 30 | 0.015 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 25 | 28 | 0.524 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 26 | 28 | 0.435 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 28 | 0.323 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 28 | 0.232 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 21 | 28 | 0.155 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 22 | 28 | 0.076 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 28 | 28 | 0.522 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 26 | 28 | 0.425 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 28 | 28 | 0.333 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 22 | 28 | 0.237 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 22 | 28 | 0.143 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 18 | 28 | 0.093 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.546 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.441 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.332 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.233 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.117 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.026 | 通过 |

## Fast-WAM · 长任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 0.5 档 | 0.4 档、0.3 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 0.2 档 | 0.1 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.3 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 93.0%，推理开销代理 0.307；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.516 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 30 | 30 | 0.440 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 30 | 30 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.234 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 30 | 30 | 0.112 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.004 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.559 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.444 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 30 | 29 | 0.326 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 28 | 29 | 0.227 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 30 | 29 | 0.111 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.006 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 30 | 28 | 0.532 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 30 | 28 | 0.440 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 29 | 28 | 0.332 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 29 | 28 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 27 | 28 | 0.131 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 26 | 28 | 0.049 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 29 | 25 | 0.534 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 28 | 25 | 0.434 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 27 | 25 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 25 | 25 | 0.241 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 24 | 25 | 0.130 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 24 | 25 | 0.035 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 25 | 28 | 0.528 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 28 | 0.327 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 25 | 28 | 0.234 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 23 | 28 | 0.137 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 20 | 28 | 0.075 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 27 | 27 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 28 | 27 | 0.426 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 26 | 27 | 0.330 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 27 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 24 | 27 | 0.134 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 20 | 27 | 0.050 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.529 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 30 | 0.440 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 30 | 0.327 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 28 | 30 | 0.236 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 26 | 30 | 0.023 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 27 | 28 | 0.526 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 28 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 28 | 0.327 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 27 | 28 | 0.232 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 24 | 28 | 0.144 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 22 | 28 | 0.077 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 25 | 28 | 0.528 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 26 | 28 | 0.422 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 28 | 28 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 20 | 28 | 0.244 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 18 | 28 | 0.165 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 18 | 28 | 0.094 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.540 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.438 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.340 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.232 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.117 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.025 | 通过 |

## Fast-WAM · 长任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 0.2 档、0.1 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 0.2 档 | 0.1 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 0.5 档 | 0.4 档、0.2 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 96.0%，推理开销代理 0.349；同任务纯推理成功率 94.0%，开销代理 1.000。全十任务纯推理成功率 94.0%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 30 | 30 | 0.518 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 30 | 30 | 0.446 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 30 | 30 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 30 | 30 | 0.235 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 30 | 30 | 0.115 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 30 | 0.008 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.556 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.443 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 30 | 29 | 0.328 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.228 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.126 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.005 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 30 | 28 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 29 | 28 | 0.438 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 30 | 28 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 28 | 28 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 29 | 28 | 0.129 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 28 | 28 | 0.049 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 28 | 25 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 27 | 25 | 0.439 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 28 | 25 | 0.343 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 28 | 25 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 23 | 25 | 0.131 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 22 | 25 | 0.038 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 27 | 28 | 0.529 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.433 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.338 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 26 | 28 | 0.243 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 24 | 28 | 0.149 | 失败 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 21 | 28 | 0.083 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 27 | 27 | 0.538 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 28 | 27 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 25 | 27 | 0.334 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 27 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 28 | 27 | 0.129 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 25 | 27 | 0.040 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 30 | 30 | 0.536 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 30 | 30 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 30 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 28 | 30 | 0.240 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 30 | 0.124 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 23 | 30 | 0.022 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 25 | 28 | 0.524 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 28 | 28 | 0.434 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 26 | 28 | 0.332 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 28 | 28 | 0.229 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 24 | 28 | 0.143 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 22 | 28 | 0.081 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 25 | 28 | 0.537 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 27 | 28 | 0.438 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 25 | 28 | 0.333 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 23 | 28 | 0.242 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 28 | 0.159 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 18 | 28 | 0.095 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.539 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.440 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.341 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.232 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.116 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.029 | 通过 |

## Fast-WAM · 空间任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.1 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 85.0%，推理开销代理 0.476；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 20 | 0.493 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.391 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.302 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 14 | 20 | 0.204 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 16 | 20 | 0.106 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 14 | 20 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.554 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.423 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.312 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.226 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.126 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.032 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.524 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 28 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.347 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 28 | 0.240 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 28 | 0.130 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.026 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.568 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.429 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 30 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 30 | 0.232 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 30 | 0.035 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 20 | 0.523 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 20 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 20 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.248 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.147 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.096 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.241 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.117 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.510 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.418 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 29 | 0.305 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.206 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 29 | 0.116 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 29 | 0.017 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 8 | 10 | 0.493 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 9 | 10 | 0.398 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 10 | 10 | 0.297 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 11 | 10 | 0.205 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 10 | 0.108 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 10 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.453 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.349 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.240 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.028 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 26 | 30 | 0.524 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.447 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 26 | 30 | 0.344 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 25 | 30 | 0.245 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 22 | 30 | 0.157 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 30 | 0.100 | 失败 |

## Fast-WAM · 空间任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.1 档 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 83.7%，推理开销代理 0.374；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.501 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 17 | 20 | 0.416 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.315 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 16 | 20 | 0.212 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 20 | 0.122 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 20 | 0.047 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.428 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.223 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.525 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 28 | 0.437 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.335 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.223 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.127 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 28 | 0.023 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.558 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.342 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 30 | 0.126 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 20 | 0.539 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 20 | 0.235 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.153 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.077 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.235 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.123 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.494 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.419 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.307 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.200 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.129 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 8 | 10 | 0.491 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 7 | 10 | 0.400 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 8 | 10 | 0.297 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 6 | 10 | 0.206 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 7 | 10 | 0.108 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 8 | 10 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.545 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.342 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.226 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.105 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 30 | 0.540 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.443 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.322 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.231 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.110 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 30 | 0.022 | 失败 |

## Fast-WAM · 空间任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.3 档、0.2 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 83.7%，推理开销代理 0.450；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.549 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.438 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 16 | 20 | 0.242 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 20 | 0.157 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 14 | 20 | 0.111 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.324 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.222 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 29 | 0.124 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.532 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.438 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 28 | 0.218 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 28 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.567 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 30 | 0.440 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.126 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 20 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.423 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 20 | 0.148 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.077 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.331 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.119 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.008 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.550 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.458 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 29 | 0.233 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.145 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.029 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 6 | 10 | 0.533 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 8 | 10 | 0.440 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 9 | 10 | 0.326 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 10 | 10 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 8 | 10 | 0.158 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 8 | 10 | 0.097 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.332 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.108 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 30 | 0.531 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.231 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.121 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 30 | 0.033 | 失败 |

## Fast-WAM · 空间任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.2 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 83.7%，推理开销代理 0.274；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.543 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.434 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 18 | 20 | 0.241 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 16 | 20 | 0.158 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 16 | 20 | 0.122 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.564 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.219 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.021 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.538 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.443 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.222 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 28 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.581 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 30 | 0.451 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.123 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.005 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 20 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.328 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 20 | 0.156 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 20 | 0.084 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.562 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.447 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.239 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.007 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.544 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.466 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 29 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.138 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.017 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 8 | 10 | 0.528 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 5 | 10 | 0.429 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 8 | 10 | 0.315 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 9 | 10 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 8 | 10 | 0.148 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 6 | 10 | 0.082 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.566 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.248 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.107 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.008 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.540 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.438 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.331 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.240 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.008 | 通过 |

## Fast-WAM · 空间任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 84.0%，推理开销代理 0.211；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.438 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 20 | 0.246 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 17 | 20 | 0.154 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 16 | 20 | 0.118 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.218 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.114 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.022 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.533 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.451 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.223 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.109 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 28 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.580 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.452 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.334 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.224 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.011 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 20 | 0.534 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 20 | 0.428 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.328 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.157 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 20 | 0.083 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.545 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.127 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.016 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 29 | 0.544 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 29 | 0.459 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 29 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.127 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.026 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 9 | 10 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 11 | 10 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 8 | 10 | 0.328 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 8 | 10 | 0.241 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 6 | 10 | 0.178 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 5 | 10 | 0.125 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.556 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.539 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.444 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.231 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 30 | 0.125 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.009 | 通过 |

## Fast-WAM · 空间任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.1 档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 83.7%，推理开销代理 0.276；同任务纯推理成功率 85.3%，开销代理 1.000。全十任务纯推理成功率 85.3%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.425 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 17 | 20 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 16 | 20 | 0.241 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 15 | 20 | 0.159 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 17 | 20 | 0.113 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.453 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.215 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.120 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.021 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.532 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 28 | 0.440 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 28 | 0.221 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.009 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.585 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.451 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.132 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 20 | 0.523 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 20 | 0.424 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.323 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 20 | 0.227 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 20 | 0.153 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.089 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.549 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.328 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.230 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.126 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.562 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.448 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.141 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.025 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 10 | 10 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 7 | 10 | 0.518 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 8 | 10 | 0.421 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 8 | 10 | 0.334 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 8 | 10 | 0.234 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 11 | 10 | 0.166 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 6 | 10 | 0.110 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.572 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.244 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.111 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.539 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.446 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.227 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.003 | 通过 |

## 汇总

| 模型 | 任务套件 | 库集数 | 已冻结任务数 | 拼接局数 | 拼接成功率 | 推理开销代理 | 同任务纯推理成功率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Fast-WAM | 长任务 | 50 | 10 | 300 | 93.3% | 0.383 | 94.0% |
| Fast-WAM | 长任务 | 100 | 10 | 300 | 94.7% | 0.371 | 94.0% |
| Fast-WAM | 长任务 | 200 | 10 | 300 | 93.0% | 0.233 | 94.0% |
| Fast-WAM | 长任务 | 300 | 10 | 300 | 93.3% | 0.269 | 94.0% |
| Fast-WAM | 长任务 | 400 | 10 | 300 | 93.0% | 0.307 | 94.0% |
| Fast-WAM | 长任务 | 500 | 10 | 300 | 96.0% | 0.349 | 94.0% |
| Fast-WAM | 空间任务 | 50 | 10 | 300 | 85.0% | 0.476 | 85.3% |
| Fast-WAM | 空间任务 | 100 | 10 | 300 | 83.7% | 0.374 | 85.3% |
| Fast-WAM | 空间任务 | 200 | 10 | 300 | 83.7% | 0.450 | 85.3% |
| Fast-WAM | 空间任务 | 300 | 10 | 300 | 83.7% | 0.274 | 85.3% |
| Fast-WAM | 空间任务 | 400 | 10 | 300 | 84.0% | 0.211 | 85.3% |
| Fast-WAM | 空间任务 | 500 | 10 | 300 | 83.7% | 0.276 | 85.3% |

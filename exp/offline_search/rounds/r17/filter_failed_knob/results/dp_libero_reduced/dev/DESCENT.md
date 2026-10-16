# 逐任务旋钮下降

开发集使用B 池每任务起点0–29，参考为同起点同一步数纯推理；主容差为5 个点，首次失败即冻结上一档。
首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。
推理开销代理为模型调用数除以执行控制数折算的决策数；DP、Fast-WAM、π0.5 和 GR00T 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。
缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。

档位数字是标定的目标 miss 比例，表中开销是实测值。

## DP · 长任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 0.3 档 | 0.2 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 0.2 档、0.1 档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 85.0%，推理开销代理 0.446；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 26 | 26 | 0.536 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.427 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 23 | 26 | 0.339 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 26 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 24 | 26 | 0.141 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 26 | 26 | 0.086 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 30 | 29 | 0.546 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 28 | 29 | 0.438 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 28 | 29 | 0.343 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.250 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 30 | 29 | 0.121 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.040 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 20 | 24 | 0.538 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 21 | 24 | 0.434 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 21 | 24 | 0.332 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 14 | 24 | 0.235 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 6 | 24 | 0.163 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 4 | 24 | 0.119 | 失败 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 22 | 15 | 0.512 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 24 | 15 | 0.400 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 22 | 15 | 0.303 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 21 | 15 | 0.202 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 23 | 15 | 0.099 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 21 | 15 | 0.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 28 | 22 | 0.546 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.438 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 22 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 26 | 22 | 0.249 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 27 | 22 | 0.132 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 24 | 22 | 0.060 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 25 | 25 | 0.511 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 25 | 25 | 0.396 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 23 | 25 | 0.301 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 24 | 25 | 0.201 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 26 | 25 | 0.101 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 20 | 25 | 0.000 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 25 | 28 | 0.529 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.425 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.327 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.234 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 28 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 28 | 28 | 0.031 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 24 | 26 | 0.511 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 25 | 26 | 0.429 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 28 | 26 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 26 | 26 | 0.244 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 28 | 26 | 0.131 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 26 | 26 | 0.080 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 20 | 21 | 0.522 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 23 | 21 | 0.418 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 16 | 21 | 0.321 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 14 | 21 | 0.228 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 13 | 21 | 0.172 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 4 | 21 | 0.143 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 29 | 0.547 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.436 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.335 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.239 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 28 | 29 | 0.107 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 29 | 29 | 0.016 | 通过 |

## DP · 长任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.4 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 90.0%，推理开销代理 0.283；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 28 | 26 | 0.521 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 27 | 26 | 0.331 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 26 | 0.227 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 24 | 26 | 0.123 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 26 | 0.012 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.539 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 28 | 29 | 0.433 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 28 | 29 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.235 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 28 | 29 | 0.126 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.012 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 27 | 24 | 0.528 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 24 | 24 | 0.422 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 25 | 24 | 0.318 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 23 | 24 | 0.220 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 21 | 24 | 0.127 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 23 | 24 | 0.050 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 22 | 15 | 0.540 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 24 | 15 | 0.440 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 26 | 15 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 28 | 15 | 0.130 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 27 | 15 | 0.027 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 28 | 22 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 26 | 22 | 0.425 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 22 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 29 | 22 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 26 | 22 | 0.126 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 28 | 22 | 0.021 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 28 | 25 | 0.544 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 26 | 25 | 0.436 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 23 | 25 | 0.326 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 23 | 25 | 0.235 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 24 | 25 | 0.128 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 24 | 25 | 0.058 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.542 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 26 | 28 | 0.428 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 30 | 28 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 28 | 0.230 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 30 | 28 | 0.114 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 28 | 0.008 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 25 | 26 | 0.499 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 26 | 0.406 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 26 | 0.308 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 26 | 26 | 0.214 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.120 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 27 | 26 | 0.037 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 18 | 21 | 0.525 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 20 | 21 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 17 | 21 | 0.327 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 18 | 21 | 0.224 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 10 | 21 | 0.144 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 9 | 21 | 0.071 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.542 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.443 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.342 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.236 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.115 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.021 | 通过 |

## DP · 长任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 91.0%，推理开销代理 0.181；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 29 | 26 | 0.515 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 25 | 26 | 0.422 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 29 | 26 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 24 | 26 | 0.237 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 25 | 26 | 0.134 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 26 | 26 | 0.045 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.536 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.428 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 29 | 29 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.113 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 29 | 29 | 0.014 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 25 | 24 | 0.540 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 25 | 24 | 0.443 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 20 | 24 | 0.340 | 失败 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 25 | 24 | 0.245 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 27 | 24 | 0.134 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 26 | 24 | 0.057 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 23 | 15 | 0.538 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 24 | 15 | 0.431 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 25 | 15 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 27 | 15 | 0.241 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 27 | 15 | 0.131 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 30 | 15 | 0.021 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 22 | 0.541 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 22 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 22 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 27 | 22 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 28 | 22 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 28 | 22 | 0.014 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 25 | 25 | 0.537 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 27 | 25 | 0.426 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 22 | 25 | 0.334 | 失败 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 26 | 25 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 25 | 25 | 0.135 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 27 | 25 | 0.047 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 28 | 28 | 0.532 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 29 | 28 | 0.424 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 27 | 28 | 0.338 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 30 | 28 | 0.115 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 28 | 28 | 0.026 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 27 | 26 | 0.507 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 27 | 26 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 27 | 26 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 26 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.132 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 27 | 26 | 0.046 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 21 | 21 | 0.530 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 23 | 21 | 0.425 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 20 | 21 | 0.325 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 17 | 21 | 0.228 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 18 | 21 | 0.147 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 15 | 21 | 0.089 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.531 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.429 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.327 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.224 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.109 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.005 | 通过 |

## DP · 长任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 91.7%，推理开销代理 0.049；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 26 | 26 | 0.526 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 26 | 0.432 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 29 | 26 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 26 | 26 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 26 | 0.123 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 26 | 0.009 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 28 | 29 | 0.543 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 30 | 29 | 0.425 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 30 | 29 | 0.329 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.236 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.118 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 28 | 29 | 0.025 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 24 | 24 | 0.535 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 26 | 24 | 0.439 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 25 | 24 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 26 | 24 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 25 | 24 | 0.134 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 26 | 24 | 0.038 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 24 | 15 | 0.538 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 21 | 15 | 0.430 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 21 | 15 | 0.335 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.243 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 24 | 15 | 0.127 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 27 | 15 | 0.022 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 26 | 22 | 0.537 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 27 | 22 | 0.431 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 22 | 0.326 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 28 | 22 | 0.237 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 29 | 22 | 0.117 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 28 | 22 | 0.013 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 25 | 25 | 0.534 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 24 | 25 | 0.425 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 24 | 25 | 0.332 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 26 | 25 | 0.234 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 27 | 25 | 0.127 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 27 | 25 | 0.025 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 28 | 28 | 0.519 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 27 | 28 | 0.427 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 27 | 28 | 0.333 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 29 | 28 | 0.234 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 28 | 28 | 0.125 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 28 | 0.017 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 26 | 0.512 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 28 | 26 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 28 | 26 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 30 | 26 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 30 | 26 | 0.118 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 27 | 26 | 0.034 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 21 | 21 | 0.525 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 22 | 21 | 0.424 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 21 | 21 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 23 | 21 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 16 | 21 | 0.145 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 16 | 21 | 0.074 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 29 | 0.528 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 28 | 29 | 0.442 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.335 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.239 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.117 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.008 | 通过 |

## DP · 长任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 0.5 档 | 0.4 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 91.3%，推理开销代理 0.191；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 27 | 26 | 0.524 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 29 | 26 | 0.442 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 26 | 26 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 29 | 26 | 0.240 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 24 | 26 | 0.141 | 失败 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 30 | 26 | 0.010 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 27 | 29 | 0.538 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.442 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 27 | 29 | 0.338 | 失败 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 28 | 29 | 0.237 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 29 | 29 | 0.121 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.011 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 23 | 24 | 0.540 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 26 | 24 | 0.435 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 24 | 24 | 0.339 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 25 | 24 | 0.241 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 25 | 24 | 0.141 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 24 | 24 | 0.063 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 24 | 15 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 22 | 15 | 0.429 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 21 | 15 | 0.337 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.239 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 26 | 15 | 0.126 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 29 | 15 | 0.016 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 29 | 22 | 0.529 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 30 | 22 | 0.428 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 28 | 22 | 0.329 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 28 | 22 | 0.235 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 29 | 22 | 0.116 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 29 | 22 | 0.013 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 26 | 25 | 0.537 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 24 | 25 | 0.424 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 25 | 25 | 0.334 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 26 | 25 | 0.240 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 26 | 25 | 0.133 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 24 | 25 | 0.039 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 28 | 28 | 0.540 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.435 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 28 | 28 | 0.328 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 27 | 28 | 0.240 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 29 | 28 | 0.123 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 29 | 28 | 0.021 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 28 | 26 | 0.510 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 28 | 26 | 0.434 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 24 | 26 | 0.327 | 失败 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 27 | 26 | 0.232 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 26 | 26 | 0.132 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 29 | 26 | 0.019 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 23 | 21 | 0.526 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 20 | 21 | 0.423 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 22 | 21 | 0.326 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 23 | 21 | 0.227 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 21 | 0.140 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 20 | 21 | 0.062 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 30 | 29 | 0.530 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 29 | 29 | 0.446 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.337 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.238 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.112 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.007 | 通过 |

## DP · 长任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.2 档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 92.0%，推理开销代理 0.147；同任务纯推理成功率 81.7%，开销代理 1.000。全任务纯推理成功率 81.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 30 | 28 | 26 | 0.523 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 30 | 26 | 26 | 0.437 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.3 档 | 30 | 28 | 26 | 0.340 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.2 档 | 30 | 28 | 26 | 0.238 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.1 档 | 30 | 28 | 26 | 0.125 | 通过 |
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 关（只有停滞门） | 30 | 29 | 26 | 0.018 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.5 档 | 30 | 29 | 29 | 0.535 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 30 | 29 | 29 | 0.423 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.3 档 | 30 | 29 | 29 | 0.330 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 30 | 30 | 29 | 0.235 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.1 档 | 30 | 30 | 29 | 0.109 | 通过 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 30 | 30 | 29 | 0.006 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 30 | 24 | 24 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 30 | 25 | 24 | 0.533 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 30 | 28 | 24 | 0.432 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.3 档 | 30 | 24 | 24 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 30 | 23 | 24 | 0.241 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.1 档 | 30 | 25 | 24 | 0.135 | 通过 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 30 | 23 | 24 | 0.060 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 30 | 15 | 15 | 1.000 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 30 | 25 | 15 | 0.536 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.4 档 | 30 | 25 | 15 | 0.436 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.3 档 | 30 | 24 | 15 | 0.336 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.2 档 | 30 | 24 | 15 | 0.238 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 30 | 26 | 15 | 0.127 | 通过 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 30 | 30 | 15 | 0.011 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 30 | 22 | 22 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.5 档 | 30 | 25 | 22 | 0.530 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.4 档 | 30 | 28 | 22 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.3 档 | 30 | 27 | 22 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.2 档 | 30 | 30 | 22 | 0.231 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 0.1 档 | 30 | 29 | 22 | 0.112 | 通过 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 关（只有停滞门） | 30 | 29 | 22 | 0.012 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 30 | 25 | 25 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 30 | 24 | 25 | 0.533 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 30 | 24 | 25 | 0.433 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 30 | 27 | 25 | 0.335 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.2 档 | 30 | 27 | 25 | 0.239 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.1 档 | 30 | 24 | 25 | 0.128 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 关（只有停滞门） | 30 | 25 | 25 | 0.039 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.5 档 | 30 | 29 | 28 | 0.528 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.4 档 | 30 | 28 | 28 | 0.430 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 30 | 29 | 28 | 0.331 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.2 档 | 30 | 30 | 28 | 0.236 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.1 档 | 30 | 27 | 28 | 0.128 | 通过 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 关（只有停滞门） | 30 | 30 | 28 | 0.006 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 30 | 26 | 26 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 30 | 25 | 26 | 0.487 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 30 | 28 | 26 | 0.417 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 30 | 29 | 26 | 0.314 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.2 档 | 30 | 28 | 26 | 0.222 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.1 档 | 30 | 27 | 26 | 0.124 | 通过 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 关（只有停滞门） | 30 | 29 | 26 | 0.021 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 30 | 21 | 21 | 1.000 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 30 | 18 | 21 | 0.534 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.4 档 | 30 | 23 | 21 | 0.418 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.3 档 | 30 | 19 | 21 | 0.325 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.2 档 | 30 | 20 | 21 | 0.223 | 通过 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.1 档 | 30 | 19 | 21 | 0.126 | 失败 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 关（只有停滞门） | 30 | 19 | 21 | 0.053 | 失败 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.5 档 | 30 | 29 | 29 | 0.535 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.4 档 | 30 | 30 | 29 | 0.443 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 30 | 30 | 29 | 0.343 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.2 档 | 30 | 30 | 29 | 0.236 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.1 档 | 30 | 30 | 29 | 0.114 | 通过 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 30 | 30 | 29 | 0.006 | 通过 |

## DP · 空间任务 · 库 50 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 0.5 档 | 0.4 档、0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.3%，推理开销代理 0.394；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 20 | 0.493 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 18 | 20 | 0.399 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 15 | 20 | 0.304 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 13 | 20 | 0.206 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 14 | 20 | 0.103 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 9 | 20 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.559 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 26 | 29 | 0.244 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 29 | 0.147 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 29 | 0.086 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.437 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.250 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.140 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.038 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.576 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.445 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.355 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.240 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.121 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.044 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 19 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 19 | 0.423 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 19 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.171 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 19 | 0.121 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 23 | 27 | 0.564 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 27 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.257 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.138 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 27 | 0.059 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.551 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 28 | 0.448 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.333 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 22 | 28 | 0.243 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 23 | 28 | 0.164 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 28 | 0.074 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 23 | 16 | 0.500 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 25 | 16 | 0.403 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 23 | 16 | 0.297 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 26 | 16 | 0.200 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 10 | 16 | 0.111 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 4 | 16 | 0.000 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 28 | 0.564 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 28 | 0.247 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.016 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.560 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 27 | 27 | 0.449 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.241 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 24 | 27 | 0.132 | 失败 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 23 | 27 | 0.076 | 失败 |

## DP · 空间任务 · 库 100 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.7%，推理开销代理 0.183；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 20 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 20 | 0.422 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 17 | 20 | 0.338 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 17 | 20 | 0.250 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 14 | 20 | 0.192 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 15 | 20 | 0.148 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.549 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.438 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 29 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 29 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 29 | 0.111 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 29 | 0.034 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.524 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 25 | 27 | 0.236 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.122 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.003 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.559 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.350 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 30 | 0.122 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 30 | 0.041 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 19 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 19 | 0.423 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 19 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 19 | 0.143 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.078 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 25 | 27 | 0.432 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.239 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.108 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.015 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.498 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.309 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 28 | 0.203 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 28 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 28 | 0.031 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 21 | 16 | 0.536 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 16 | 0.427 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 26 | 16 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 16 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 16 | 0.130 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 16 | 0.040 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.566 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.458 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 28 | 0.241 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 28 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.018 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.529 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.437 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.336 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 27 | 0.239 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 27 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 27 | 0.041 | 通过 |

## DP · 空间任务 · 库 200 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.7%，推理开销代理 0.115；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.535 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 18 | 20 | 0.346 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.250 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 20 | 0.164 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.112 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 29 | 0.577 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.321 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 29 | 0.223 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.134 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 29 | 0.020 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.525 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.122 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.586 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.446 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 27 | 30 | 0.347 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.226 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.109 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 30 | 0.005 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 19 | 0.553 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 19 | 0.335 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 19 | 0.157 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 19 | 0.083 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.565 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.453 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.115 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.023 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.535 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.331 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.221 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.116 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.025 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 22 | 16 | 0.533 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 16 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 16 | 0.334 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 16 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 16 | 0.146 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 16 | 0.043 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.114 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.010 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.545 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.444 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.237 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 27 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 27 | 0.024 | 通过 |

## DP · 空间任务 · 库 300 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 0.1 档 | 关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 关（只有停滞门） | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 87.7%，推理开销代理 0.064；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.538 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 20 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.255 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 18 | 20 | 0.163 | 失败 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 20 | 0.115 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.557 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 29 | 0.433 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.327 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.231 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.117 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 29 | 0.036 | 失败 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.524 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 27 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.231 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.012 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.579 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 30 | 0.448 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 30 | 0.344 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 30 | 0.234 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.105 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 19 | 0.530 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 19 | 0.427 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 19 | 0.339 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 20 | 19 | 0.255 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.155 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 18 | 19 | 0.086 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 26 | 27 | 0.572 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.450 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.232 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.113 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 27 | 0.021 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.543 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 28 | 0.450 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 28 | 0.326 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.200 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.102 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 22 | 16 | 0.516 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 22 | 16 | 0.422 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 16 | 0.335 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 16 | 0.243 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 16 | 0.149 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 16 | 0.067 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.542 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 28 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.238 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.106 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.001 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.531 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 27 | 0.432 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.321 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.233 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 27 | 0.114 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 27 | 0.025 | 通过 |

## DP · 空间任务 · 库 400 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 0.2 档 | 0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 0.3 档 | 0.2 档、0.1 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 89.3%，推理开销代理 0.106；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 20 | 0.549 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 20 | 0.434 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 20 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.245 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 19 | 20 | 0.164 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 20 | 0.120 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.565 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.435 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 29 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 27 | 29 | 0.242 | 失败 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 29 | 0.122 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.035 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 27 | 27 | 0.530 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.346 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.228 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 27 | 0.121 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 30 | 0.598 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 30 | 0.436 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 30 | 0.339 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 30 | 0.227 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 30 | 0.103 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.013 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 19 | 19 | 0.548 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 19 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 21 | 19 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.241 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 19 | 0.157 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 19 | 19 | 0.085 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.568 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.337 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 27 | 0.230 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.125 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 27 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.538 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.204 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.103 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.002 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 16 | 0.516 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 16 | 0.441 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 16 | 0.340 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 25 | 16 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 27 | 16 | 0.142 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 16 | 0.055 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.236 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.006 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.516 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 27 | 0.425 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 27 | 0.322 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.222 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 26 | 27 | 0.112 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 26 | 27 | 0.030 | 通过 |

## DP · 空间任务 · 库 500 集

| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |
|---|---|---|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 0.4 档 | 0.3 档、关（只有停滞门） | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 无 | 无 | 完整 |

已冻结 10/10 个任务；拼接 300 局，成功率 88.7%，推理开销代理 0.091；同任务纯推理成功率 83.7%，开销代理 1.000。全任务纯推理成功率 83.7%。

| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |
|---|---|---:|---:|---:|---:|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 20 | 20 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 20 | 20 | 0.527 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 19 | 20 | 0.439 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 20 | 20 | 0.338 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 19 | 20 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 20 | 20 | 0.157 | 通过 |
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 20 | 0.110 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 30 | 29 | 29 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 29 | 0.555 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 29 | 0.445 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 29 | 0.325 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 29 | 0.227 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 29 | 0.127 | 通过 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 28 | 29 | 0.032 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.523 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.442 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.230 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.119 | 通过 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.012 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 30 | 30 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 30 | 0.583 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 28 | 30 | 0.430 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 30 | 0.343 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 30 | 0.229 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 28 | 30 | 0.127 | 失败 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 30 | 0.012 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 30 | 19 | 19 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 30 | 18 | 19 | 0.537 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 30 | 20 | 19 | 0.423 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.3 档 | 30 | 19 | 19 | 0.334 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 30 | 21 | 19 | 0.242 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.1 档 | 30 | 21 | 19 | 0.155 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 20 | 19 | 0.082 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 27 | 0.568 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 29 | 27 | 0.449 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 28 | 27 | 0.345 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 28 | 27 | 0.235 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.119 | 通过 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 27 | 0.014 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.5 档 | 30 | 29 | 28 | 0.547 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.452 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 28 | 0.320 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.201 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.090 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 29 | 28 | 0.017 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 30 | 16 | 16 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 30 | 22 | 16 | 0.518 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 30 | 21 | 16 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 30 | 22 | 16 | 0.331 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.2 档 | 30 | 24 | 16 | 0.244 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 30 | 24 | 16 | 0.155 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 25 | 16 | 0.062 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 纯推理 | 30 | 28 | 28 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 30 | 30 | 28 | 0.552 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 28 | 0.429 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.3 档 | 30 | 29 | 28 | 0.341 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.2 档 | 30 | 30 | 28 | 0.225 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 30 | 30 | 28 | 0.118 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 30 | 28 | 0.004 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 30 | 27 | 27 | 1.000 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 30 | 28 | 27 | 0.522 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 30 | 30 | 27 | 0.430 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.3 档 | 30 | 30 | 27 | 0.326 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.2 档 | 30 | 29 | 27 | 0.222 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 30 | 29 | 27 | 0.110 | 通过 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 30 | 27 | 27 | 0.021 | 通过 |

## 汇总

| 模型 | 任务套件 | 库集数 | 已冻结任务数 | 拼接局数 | 拼接成功率 | 推理开销代理 | 同任务纯推理成功率 |
|---|---|---:|---:|---:|---:|---:|---:|
| DP | 长任务 | 50 | 10 | 300 | 85.0% | 0.446 | 81.7% |
| DP | 长任务 | 100 | 10 | 300 | 90.0% | 0.283 | 81.7% |
| DP | 长任务 | 200 | 10 | 300 | 91.0% | 0.181 | 81.7% |
| DP | 长任务 | 300 | 10 | 300 | 91.7% | 0.049 | 81.7% |
| DP | 长任务 | 400 | 10 | 300 | 91.3% | 0.191 | 81.7% |
| DP | 长任务 | 500 | 10 | 300 | 92.0% | 0.147 | 81.7% |
| DP | 空间任务 | 50 | 10 | 300 | 89.3% | 0.394 | 83.7% |
| DP | 空间任务 | 100 | 10 | 300 | 89.7% | 0.183 | 83.7% |
| DP | 空间任务 | 200 | 10 | 300 | 89.7% | 0.115 | 83.7% |
| DP | 空间任务 | 300 | 10 | 300 | 87.7% | 0.064 | 83.7% |
| DP | 空间任务 | 400 | 10 | 300 | 89.3% | 0.106 | 83.7% |
| DP | 空间任务 | 500 | 10 | 300 | 88.7% | 0.091 | 83.7% |

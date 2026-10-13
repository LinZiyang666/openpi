# 逐任务旋钮测试

使用 A 集官方起点 0–49，每任务固定一个档位，与同起点纯推理对照配对。
纯推理对照为DP DDIM10 eta0、Fast-WAM K1、Cosmos K2含清理步（NFE1）的10-10 A 集运行；按逐任务提档表复测，迭代到每个任务差距不超过4 个点。
推理开销代理按模型调用数除以执行控制数折算的决策数计算；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次，纯推理按一计。
成功率差的95% 区间在每个任务内按起点配对重采样5,000 次，随机种子为20261005。
任务成功率低于对照至少 10 个点，提到纯推理；低 6–9 个点，提一档，0.5 档提到纯推理；纯推理保持。阈值用成功局数作整数比较。
缺局只报告已有配对，不补齐、不按部分结果提档；缺少完整执行计数时开销记为暂无。提档清单供后续复测，当前结果仍是提档前的测量。

| 模型 | 任务套件 | 库集数 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 95% 区间（个点） | 新增成功 | 丢失成功 | 推理开销代理 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| Cosmos | 长任务 | 50 | 500 | 500 | 96.2% | 96.6% | -0.4 | [-1.6, +0.8] | 3 | 5 | 0.953 | 完整 |
| Cosmos | 长任务 | 100 | 500 | 500 | 95.6% | 96.6% | -1.0 | [-2.4, +0.4] | 4 | 9 | 0.749 | 完整 |
| Cosmos | 长任务 | 200 | 500 | 500 | 96.2% | 96.6% | -0.4 | [-1.8, +1.0] | 6 | 8 | 0.738 | 完整 |
| Cosmos | 长任务 | 300 | 500 | 500 | 96.0% | 96.6% | -0.6 | [-2.8, +1.4] | 13 | 16 | 0.484 | 完整 |
| Cosmos | 长任务 | 400 | 500 | 500 | 95.8% | 96.6% | -0.8 | [-2.4, +0.8] | 6 | 10 | 0.714 | 完整 |
| Cosmos | 长任务 | 500 | 500 | 500 | 96.2% | 96.6% | -0.4 | [-2.4, +1.6] | 11 | 13 | 0.459 | 完整 |
| Cosmos | 空间任务 | 50 | 500 | 500 | 96.0% | 96.6% | -0.6 | [-2.6, +1.4] | 12 | 15 | 0.809 | 完整 |
| Cosmos | 空间任务 | 100 | 500 | 500 | 96.2% | 96.6% | -0.4 | [-2.6, +1.8] | 14 | 16 | 0.462 | 完整 |
| Cosmos | 空间任务 | 200 | 500 | 500 | 98.0% | 96.6% | +1.4 | [-0.2, +3.0] | 12 | 5 | 0.382 | 完整 |
| Cosmos | 空间任务 | 300 | 500 | 500 | 96.8% | 96.6% | +0.2 | [-1.8, +2.0] | 13 | 12 | 0.144 | 完整 |
| Cosmos | 空间任务 | 400 | 500 | 500 | 96.6% | 96.6% | +0.0 | [-1.4, +1.4] | 7 | 7 | 0.250 | 完整 |
| Cosmos | 空间任务 | 500 | 500 | 500 | 96.6% | 96.6% | +0.0 | [-2.0, +2.0] | 13 | 13 | 0.104 | 完整 |

## Cosmos · 长任务 · 库 50 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 50 | 50 | 94.0% | 94.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.5 档 | 50 | 50 | 88.0% | 92.0% | -4.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 长任务 · 库 100 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 50 | 50 | 94.0% | 94.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.5 档 | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 50 | 50 | 94.0% | 98.0% | -4.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 0.3 档 | 50 | 50 | 94.0% | 98.0% | -4.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 0.3 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |

## Cosmos · 长任务 · 库 200 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 纯推理 | 50 | 50 | 94.0% | 94.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.4 档 | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.5 档 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |

## Cosmos · 长任务 · 库 300 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.5 档 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 关（只有停滞门） | 50 | 50 | 98.0% | 94.0% | +4.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 0.2 档 | 50 | 50 | 92.0% | 96.0% | -4.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 关（只有停滞门） | 50 | 50 | 90.0% | 92.0% | -2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.5 档 | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 0.5 档 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |

## Cosmos · 长任务 · 库 400 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 50 | 50 | 96.0% | 100.0% | -4.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.4 档 | 50 | 50 | 92.0% | 94.0% | -2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 纯推理 | 50 | 50 | 94.0% | 92.0% | +2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.3 档 | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.4 档 | 50 | 50 | 94.0% | 96.0% | -2.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 长任务 · 库 500 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 0.4 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 0.2 档 | 50 | 50 | 94.0% | 94.0% | +0.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 关（只有停滞门） | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 0.1 档 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 0.4 档 | 50 | 50 | 94.0% | 98.0% | -4.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 0.3 档 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 纯推理 | 50 | 50 | 92.0% | 92.0% | +0.0 | 无需提档 | 完整 |
| libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 关（只有停滞门） | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 50 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 94.0% | 96.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 84.0% | 84.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 94.0% | 96.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 100 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 92.0% | 96.0% | -4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 96.0% | 100.0% | -4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 88.0% | 84.0% | +4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 96.0% | 100.0% | -4.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 200 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 90.0% | 84.0% | +6.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0.1 档 | 50 | 50 | 100.0% | 96.0% | +4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 300 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.1 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 96.0% | 98.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 92.0% | 96.0% | -4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 96.0% | 96.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 90.0% | 84.0% | +6.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 400 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0.4 档 | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 50 | 50 | 92.0% | 96.0% | -4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 96.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 纯推理 | 50 | 50 | 84.0% | 84.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 96.0% | +4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0.1 档 | 50 | 50 | 96.0% | 100.0% | -4.0 | 无需提档 | 完整 |

## Cosmos · 空间任务 · 库 500 集

| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |
|---|---|---:|---:|---:|---:|---:|---|---|
| libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0.3 档 | 50 | 50 | 100.0% | 100.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 98.0% | +0.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 98.0% | +2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0.2 档 | 50 | 50 | 92.0% | 96.0% | -4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 98.0% | 100.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 96.0% | +4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0.5 档 | 50 | 50 | 82.0% | 84.0% | -2.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 100.0% | 96.0% | +4.0 | 无需提档 | 完整 |
| libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 关（只有停滞门） | 50 | 50 | 96.0% | 100.0% | -4.0 | 无需提档 | 完整 |


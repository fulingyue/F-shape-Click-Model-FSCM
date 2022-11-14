import numpy as np
import os

row_num = [
    [5, 3],
    [3, 3]
] # row num

C_num = [[3,3,3,3,3,6,6,6,6,6],[3,3,3,3,3,6,6,6,6,6]]
R_num = [[5,5,5,5,5,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3]]
R_length = [[10,10,10,10,10,8,8,8,8,8],[8,8,8,8,8,8,8,8,8,8]]

C_pos_num = [[18,24],[11,18]]
pos_fixation_C = np.zeros(27)
pos_fixation_R = np.zeros([3,8])


B_num = [[11,7],[7,7]]
dict_list = [
    [{0:0, 1:0, 2:0, 3:1,
      4:2, 5:2, 6:2, 7:3,
      8:4, 9:4, 10:4, 11:5,
      12:6,13:6,14:6,15:7,
      16:8,17:8,18:8,19:9,
      20:10,21:10,22:10},
    {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:1,
     7:2, 8:2, 9:2,10:2,11:2,12:2, 13:3,
     14:4,15:4,16:4,17:4,18:4,19:4,20:5,
     21:6,22:6,23:6,24:6,25:6,26:6}],
    [{0:0, 1:0, 2:1,
      3:2,4:2,5:2, 6:3,
      7:4,8:4,9:4,10:5,
      11:6,12:6,13:6},
    {0:0, 1:0, 2:1,
     3:2,4:2,5:2,6:2,7:2,8:2, 9:3,
     10:4,11:4,12:4,13:4,14:4,15:4, 16:5,
     17:6,18:6,19:6,20:6,21:6,22:6}]
]

def seq(x,y):
    if x[0] < y[0]:
        return True
    if x[0] == y[0] and x[1] < y[1]:
        return True
    return False


def jump_type(prev, cur, nxt, dict):
    # intra-block:0, inter-block 1
    # seq: 0, compare: 1, others: 2
    prev_block_num = dict[prev[0]]
    cur_block_num = dict[cur[0]]
    next_block_num = dict[nxt[0]]

    if prev_block_num == cur_block_num and cur_block_num == next_block_num:
        # intra-block
        if prev_block_num % 2 == 0:
            # V
            if prev[0] < cur[0] and cur[0] < nxt[0]:
                return 0, 0 # 123
            elif (prev == nxt).all():
                if prev[0] < cur[0]:
                    return 0,1 # 121
                else:
                    return 0,2 # 212
            else:
                if prev[0] > cur[0] and cur[0] > nxt[0]:
                    return 0,3 # 321
                elif prev[0] > cur[0] and cur[0] < nxt[0] and prev[0] > nxt[0]:
                    return 0,4 # 312
                elif prev[0] > cur[0] and cur[0] < nxt[0] and prev[0] < nxt[0]:
                    return 0,5 # 213
                elif prev[0] < cur[0] and cur[0] > nxt[0]and prev[0] > nxt[0]:
                    return 0,6 # 231
                elif prev[0] < cur[0] and cur[0] > nxt[0]and prev[0] < nxt[0]:
                    return 0,7 # 132

        else:
            # H
            assert prev[0] == cur[0]
            assert prev[0] == nxt[0]
            if prev[1] < cur[1] and cur[1] < nxt[1]:
                return 0,0
            elif prev[1] == nxt[1]:
                if prev[1] < cur[1]:
                    return 0, 1  # 121
                else:
                    return 0, 2  # 212
            else:
                if prev[1] > cur[1] and cur[1] > nxt[1]:
                    return 0, 3  # 321
                elif prev[1] > cur[1] and cur[1] < nxt[1] and prev[1] > nxt[1]:
                    return 0, 4  # 312
                elif prev[1] > cur[1] and cur[1] < nxt[1] and prev[1] < nxt[1]:
                    return 0, 5  # 213
                elif prev[1] < cur[1] and cur[1] > nxt[1] and prev[1] > nxt[1]:
                    return 0, 6  # 231
                elif prev[1] < cur[1] and cur[1] > nxt[1] and prev[1] < nxt[1]:
                    return 0, 7  # 132
    else:
        # inter-block
        if seq(prev, cur) and seq(cur, nxt):
            return 1,0
        elif seq(cur,prev) and seq(nxt,cur):
            return 1,3
        elif seq(prev, cur) and seq(nxt, cur):
            if seq(prev, nxt):
                return 1,7#132
            elif seq(nxt, prev):
                return 1,6 #231
            else:
                return 1,1 #121
        else:
            if seq(prev,nxt):
                return 1,5#213
            elif seq(nxt, prev):
                return 1,4#312
            else:
                return 1,2




def dealE2(path):
    file = os.path.join('eyetracking_data/exam_seqs', path)
    strs = path.strip('.npy').strip('exp').split('_')
    exp_num = int(strs[0])
    page_num = int(strs[1])
    exp_type = page_num // 5
    dict = dict_list[exp_num][exp_type]
    exam_list = np.load(file)
    type_num = np.zeros([2,8])
    block_skip = np.zeros(7)
    block_skip_col_src = np.zeros(6)
    block_skip_col_dest = np.zeros(6)

    block_skip_row_src = np.zeros(8)
    block_skip_row_dest = np.zeros(8)

    block_skip_col, block_skip_row = 0,0
    if exp_num == 0:
        bias = 0
    else:
        bias = 3

    for idx, item in enumerate(exam_list):
        if idx == 0 or idx == len(exam_list)-1:
            continue
        else:
            x,y = jump_type(exam_list[idx-1], item, exam_list[idx+1],dict)
            type_num[x][y] += 1

    for idx, item in enumerate(exam_list):
        if idx == 0:
            continue
        else:
            prev = exam_list[idx-1]
            cur = item

            prev_b = dict[prev[0]]
            cur_b = dict[cur[0]]

            if cur_b - prev_b > 0:
                block_skip[cur_b - prev_b] += 1

            if cur_b - prev_b == 2:
                if cur_b % 2 == 0:
                    block_skip_col += 1
                if page_num > 5 and cur_b % 2 == 0:
                    # V
                    block_skip_col_src[(prev[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
                    block_skip_col_dest[(cur[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
                else:
                    if prev[1] >= 0 and cur[1] >= 0:
                        block_skip_row += 1
                    if prev[1] < 8 and prev[1] >= 0:
                        block_skip_row_src[prev[1]] += 1
                    if cur[1] < 8 and cur[1] >= 0:
                        block_skip_row_dest[cur[1]] += 1

    return type_num, [block_skip, block_skip_col, block_skip_row], block_skip_col_src, block_skip_col_dest, block_skip_row_src, block_skip_row_dest


def dealE(path):
    file = os.path.join('eyetracking_data/exam_seqs', path)
    strs = path.strip('.npy').strip('exp').split('_')
    exp_num = int(strs[0])
    page_num = int(strs[1])
    exp_type = page_num // 5
    dict = dict_list[exp_num][exp_type]

    exam_list = np.load(file) # N,2

    inter_block = 0
    outer_block = 0
    in_order = 0


    outer_block_from_r2c = 0
    outer_block_from_c2r = 0
    outer_block_from_c2c = 0
    outer_block_from_r2r = 0
    outer_col_dest_from_col = np.zeros(C_num[exp_num][page_num])
    outer_col_dest_from_row = np.zeros(C_num[exp_num][page_num])

    interB_row_dest = np.zeros(10)
    interB_col_dest = np.zeros(C_num[exp_num][page_num])
    interB_row_src = np.zeros(10)
    interB_col_src = np.zeros(C_num[exp_num][page_num])

    window_length_col = np.zeros(C_num[exp_num][page_num])
    window_length_row = np.zeros(10)

    if exp_num == 0:
        bias = 0
    else:
        bias = 3

    for idx,item in enumerate(exam_list):
        if idx == 0:
            continue
        post = exam_list[idx -1]

        post_type = dict[post[0]] % 2
        item_type = dict[item[0]] % 2
        # if post_type == 1 and post[1] == -1:
        #     continue
        # if item_type == 1 and item[1] == -1:
        #     continue

        if item[0] > post[0]: # in order
            in_order += 1

        else:
            # in order?
            if post_type + item_type == 1:
                # V-H, H-V
                outer_block += 1
                if post_type == 1:
                    assert item_type == 0
                    # H-V
                    outer_block_from_r2c += 1
                    if exp_num == 1 and item[0] < 2:
                       pass
                    else:
                        outer_col_dest_from_row[(item[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
                else:
                    assert item_type == 1
                    outer_block_from_c2r += 1

            else:
                if dict[item[0]] == dict[post[0]]:
                    # inter block revisit
                    inter_block += 1
                    if post_type == 1:
                        # in H
                        if item[1] > post[1]:
                            in_order += 1
                            inter_block -= 1
                        else:
                            interB_row_src[post[1]] += 1
                            interB_row_dest[item[1]] += 1
                            window_length_row[post[1] - item[1]] += 1
                    else:
                        # in V
                        window_length_col[post[0] - item[0]] += 1
                        if exp_num == 1 and item[0] < 2:
                            interB_col_src[post[0]] += 1
                            interB_col_dest[item[0]] += 1
                        else:
                            interB_col_src[(post[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
                            interB_col_dest[(item[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
                else:
                    outer_block += 1
                    #H-H
                    if post_type == 1:
                        assert item_type == 1
                        outer_block_from_r2r += 1
                    else:
                        # V-V
                        assert item_type == 0
                        outer_block_from_c2c += 1
                        outer_col_dest_from_col[(item[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1



    jump_sum = inter_block + outer_block + in_order

    interB_col_src_sum = interB_col_src.sum()
    interB_row_src_sum = interB_row_src.sum()

    interB_col_dest_sum = interB_col_dest.sum()
    interB_row_dest_sum = interB_row_dest.sum()

    if interB_col_dest_sum == 0:
        interB_col_dest_sum = 1
    if interB_row_dest_sum == 0:
        interB_row_dest_sum = 1
    if interB_col_src_sum == 0:
        interB_col_src_sum = 1
    if interB_row_src_sum == 0:
        interB_row_src_sum = 1

    outer_col_dest_sum = outer_col_dest_from_row.sum() + outer_col_dest_from_col.sum()
    if outer_col_dest_sum == 0:
        outer_col_dest_sum = 1


    B_num_i = B_num[exp_num][exp_type]
    B_flag = np.zeros(B_num_i)
    for idx, item in enumerate(exam_list):
        B_flag[dict[item[0]]] = 1

    jump_num = B_num_i - B_flag.sum()
    B_flag_col = B_num_i // 2 + 1 - B_flag[list(range(0, B_num_i,2))].sum()
    B_flag_row = B_num_i // 2 - B_flag[list(range(1, B_num_i, 2))].sum()
    jump_divisor = jump_num
    if jump_num == 0:
        jump_divisor = 1

    window_length_col_sum = window_length_col.sum()
    if window_length_col_sum == 0:
        window_length_col_sum = 1
    window_length_row_sum = window_length_row.sum()
    if window_length_row_sum == 0:
        window_length_row_sum = 1

    return exp_num, page_num, \
           inter_block / jump_sum , outer_block / jump_sum, in_order / jump_sum, \
           interB_row_src/ interB_row_src_sum, interB_col_src / interB_col_src_sum, \
           interB_row_dest / interB_row_dest_sum, interB_col_dest / interB_col_dest_sum, \
           outer_block_from_c2r / jump_sum, outer_block_from_r2c / jump_sum,  \
           outer_block_from_c2c / jump_sum, outer_block_from_r2r / jump_sum, \
           jump_num/B_num_i, B_flag_col/jump_divisor, B_flag_row/jump_divisor, \
           window_length_row/window_length_row_sum, window_length_col/window_length_col_sum,\
           outer_col_dest_from_col /outer_col_dest_sum, outer_col_dest_from_row/outer_col_dest_sum


def dealF(path):

    file = os.path.join('eyetracking_data/fixation_list', path)
    strs = path.strip('.npy').strip('exp').split('_')
    exp_num = int(strs[0])
    page_num = int(strs[1])
    exp_type = page_num // 5

    # RQ1
    B_fixation = np.zeros(B_num[exp_num][exp_type])
    dict = dict_list[exp_num][exp_type]

    # RQ2
    col_fixation = [0] * C_num[exp_num][page_num]
    row_fixation = [0] * 10
    fixation_list = np.load(file)  # N,2


    if exp_num == 0 and exp_type == 1:
        for item in fixation_list:
            x = item[0]
            y = item[1]
            if y != -1 and x % 7 == 6: #row
                pos_fixation_R[x // 7][y] += 1
            elif x % 7 != 6:
                pos_fixation_C[x] += 1



    if fixation_list.shape[0] <= 10:
        return 0,0, np.zeros(3),np.zeros(10), np.zeros(11)
    if exp_num == 0:
        bias = 0
    else:
        bias = 3

    for item in fixation_list:
        # RQ1
        if item[0]!= -1:
            t = dict[item[0]] % 2
            if t == 0:
                # V
                B_fixation[dict[item[0]]] += 1
                if exp_num == 1 and item[0] < 2:
                    continue
                    # col_fixation[item[0]] += 1

                else:
                    # if (item[0] - bias) // (C_num[exp_num][page_num] + 1)
                    col_fixation[(item[0] - bias) % (C_num[exp_num][page_num] + 1)] += 1
            else:
                # H
                if item[1] != -1:
                    B_fixation[dict[item[0]]] += 1
                    row_fixation[item[1]] += 1



    col_fixation = np.array(col_fixation)
    row_fixation = np.array(row_fixation)

    if np.sum(col_fixation) < 1:
        return exp_num, exp_type,\
           np.zeros(col_fixation.shape), \
           row_fixation / np.sum(row_fixation), \
           B_fixation / np.sum(B_fixation)

    if np.sum(row_fixation) < 1:
        return exp_num, exp_type, \
               col_fixation / np.sum(col_fixation), \
               np.zeros(10), \
               B_fixation / np.sum(B_fixation)

    return exp_num, exp_type,\
           col_fixation / np.sum(col_fixation), \
           row_fixation / np.sum(row_fixation), \
           B_fixation / np.sum(B_fixation)


def find_first(array, lower, higher):
    idxs = np.where((array >= lower) &(array <= higher))
    return idxs[0]


if __name__ == '__main__':
    path = 'eyetracking_data/'
    interB = 0
    outerB = 0
    in_orderB = 0

    colF6 = np.zeros(6)
    colF3 = np.zeros(3)
    rowF = np.zeros(10)
    fixB7 = np.zeros(7)
    fixB = [[np.zeros(11),np.zeros(7)], [np.zeros(7), np.zeros(7)]]
    fixB11 = np.zeros(11)

    interB_C3_src = np.zeros(3)
    interB_C6_src = np.zeros(6)
    interB_R_src = np.zeros(10)

    interB_C3_dest = np.zeros(3)
    interB_C6_dest = np.zeros(6)
    interB_R_dest = np.zeros(10)

    outerB_C2R = []
    outerB_R2C = []
    outerB_C2C = []
    outerB_R2R = []

    outerB_C3_dest_from_col = np.zeros(3)
    outerB_C6_dest_from_col = np.zeros(6)
    outerB_C3_dest_from_row = np.zeros(3)
    outerB_C6_dest_from_row = np.zeros(6)

    jump_num_list = []
    jump_col_list = []
    jump_row_list = []


    window_R = np.zeros(10)
    window_C3 = np.zeros(3)
    window_C6 = np.zeros(6)

    block_skip = np.zeros(7)
    block_skip_col = 0
    block_skip_row = 0
    block_skip_col_src = np.zeros(6)
    block_skip_col_dest = np.zeros(6)
    block_skip_row_src = np.zeros(8)
    block_skip_row_dest = np.zeros(8)

    comparision = np.zeros([2,8])
    for paths in os.listdir(os.path.join(path,'exam_seqs')):
        if paths == '.DS_Store' or paths == '1':
            continue
        exp_num, page_num, inter_block, outer_block, in_order, \
        inter_row_src, inter_col_src,inter_row_dest, inter_col_dest , outer_c2r, outer_r2c, outer_c2c, outer_r2r, \
            jump_num, jump_col, jump_row, window_row, window_col, outer_col_dest_from_col, outer_col_dest_from_row = dealE(paths)

        comp_i, block_skip_i, block_skip_col_src_i, block_skip_col_dest_i, block_skip_row_src_i, block_skip_row_dest_i = dealE2(paths)
        block_skip += block_skip_i[0]
        block_skip_col += block_skip_i[1]
        block_skip_row += block_skip_i[2]
        block_skip_col_src += block_skip_col_src_i
        block_skip_col_dest += block_skip_col_dest_i
        block_skip_row_src += block_skip_row_src_i
        block_skip_row_dest += block_skip_row_dest_i
        comparision += comp_i

        if outer_col_dest_from_row.shape[0] ==3:
            outerB_C3_dest_from_row += outer_col_dest_from_row
            outerB_C3_dest_from_col += outer_col_dest_from_col
        else:
            outerB_C6_dest_from_row+= outer_col_dest_from_row
            outerB_C6_dest_from_col += outer_col_dest_from_col

        window_R += window_row
        if window_col.shape[0] == 3:
            window_C3 += window_col
        else:
            window_C6 += window_col


        jump_num_list.append(jump_num)
        if jump_col != 0 or jump_row != 0:
            jump_col_list.append(jump_col)
            jump_row_list.append(jump_row)

        in_orderB += in_order
        interB += inter_block
        outerB += outer_block
        if C_num[exp_num][page_num] ==3:
            interB_C3_src += inter_col_src
            interB_C3_dest += inter_col_dest
        else:
            interB_C6_src += inter_col_src
            interB_C6_dest += inter_col_dest

        interB_R_dest += inter_row_dest
        interB_R_src += inter_row_src
        outerB_C2R.append(outer_c2r)
        outerB_R2C.append(outer_r2c)
        outerB_C2C.append(outer_c2c)
        outerB_R2R.append(outer_r2r)


    num = 0


    for paths in os.listdir(os.path.join(path, 'fixation_list')):
        if paths == '.DS_Store'or paths == '1':
            continue
        num += 1
        exp_num, exp_type, col_fixation,row_fixation, B_fixation = dealF(paths)
        if len(col_fixation) == 3:
            colF3 += col_fixation
        else:
            colF6 += col_fixation
        rowF += row_fixation

        fixB[exp_num][exp_type] += B_fixation
        if B_fixation.shape[0] == 7:
            fixB7 += B_fixation
        else:
            fixB11 += B_fixation



    sumB = interB + outerB + in_orderB

    pos_sum = np.sum(pos_fixation_C) + np.sum(pos_fixation_R)
    print('fixation pos C', pos_fixation_C/pos_sum)
    print('fixation pos R', pos_fixation_R/pos_sum)

    print('RQ1:')
    print(fixB[0][0] / fixB[0][0].sum())
    print(fixB[0][1] / fixB[0][1].sum())
    print(fixB[1][0] / fixB[1][0].sum())
    print(fixB[1][1] / fixB[1][1].sum())
    # print(fixB7 / fixB7.sum())
    print(fixB11 / fixB11.sum())
    print('RQ2:')
    print('col3:', colF3 / colF3.sum())
    print('col6:', colF6 / colF6.sum())
    print('row', rowF / rowF.sum())

    print('RQ3:')
    print('inter block revisit rate:',interB/sumB)
    print('outer block revisit rate:',outerB/sumB)
    print('in order block visit rate:',in_orderB/ sumB)

    print('inter block revisit position of R(src)', interB_R_src / interB_R_src.sum())
    print('inter block revisit position of C3(src)', interB_C3_src/interB_C3_src.sum())
    print('inter block revisit position of C6(src)', interB_C6_src / interB_C6_src.sum())
    print('inter block revisit position of R(dest)', interB_R_dest / interB_R_dest.sum())
    print('inter block revisit position of C3(dest)', interB_C3_dest/interB_C3_dest.sum())
    print('inter block revisit position of C6(dest)', interB_C6_dest / interB_C6_dest.sum())

    print('window_size rate of row:', window_R / window_R.sum())
    print('window size rate of col3:', window_C3 / window_C3.sum())
    print('window size rate of col6:', window_C6 / window_C6.sum())

    print('outer block revisit of R2C', np.average(np.array(outerB_R2C)))
    print('outer block revisit of C2R', np.average(np.array(outerB_C2R)))
    print('outer block revisit of C2C', np.average(np.array(outerB_C2C)))
    print('outer block revisit of R2R', np.average(np.array(outerB_R2R)))

    print('outer block revisit col3 dest from row', outerB_C3_dest_from_row / (outerB_C3_dest_from_row.sum() +  outerB_C3_dest_from_col.sum()))
    print('outer block revisit col3 dest from col',
          outerB_C3_dest_from_col / (outerB_C3_dest_from_col.sum() + outerB_C3_dest_from_row.sum()))

    print('outer block revisit col6 dest from row', outerB_C6_dest_from_row / (outerB_C6_dest_from_row.sum()+ outerB_C6_dest_from_col.sum()))
    print('outer block revisit col6 dest from col', outerB_C6_dest_from_col / (outerB_C6_dest_from_col.sum()+outerB_C6_dest_from_row.sum()))

    print('COmparision type distribution', comparision)
    print('comparison without inter-intra', comparision.sum(axis=0))

    print('RQ4:')
    print('jump num rate in num blocks:', np.average(np.array(jump_num_list)))
    print('jump col rate in jump num:', np.average(np.array(jump_col_list)))
    print('jump row rate in jump num:', np.average(np.array(jump_row_list)))
    block_skip[1] = 0
    print('block skip window size:', block_skip )
    print('block skip col, row :', block_skip_col, block_skip_row)
    print('block skip col src:', block_skip_col_src / np.sum(block_skip_col_src))
    print('block skip col dest:', block_skip_col_dest / np.sum(block_skip_col_dest))

    print('block skip row src:', block_skip_row_src / np.sum(block_skip_row_src))
    print('block skip row dest:', block_skip_row_dest / np.sum(block_skip_row_dest))



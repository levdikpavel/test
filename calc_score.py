import pandas as pd
import os

OBJ_REAL_DIR = 'stuff'
OBJ_REAL_FILE = 'objects_train.csv'
IOU_THRESHOLD = 0.5
CLASSES = [1, 2, 3]


def read(filepath):
    sep = ';'
    """
    f = open(filepath, 'r')
    first_row_str = f.readline()
    f.close()
    if first_row_str.find(';') == -1:
        sep = ','
    print first_row_str
    """
    df = pd.read_csv(filepath, sep=sep, header=0)
    #"""
    columns = list(df.columns)
    if 's' not in columns:
        df['s'] = 1
    #"""
    # print df.columns
    # print list(df)

    # df = df.rename(columns={list(df)[0]: 'img_id'})
    df = df.rename(columns={list(df)[1]: 'bb_coord'})
    df = df.rename(columns={list(df)[2]: 'class'})
    return df


def calc_iou(coord1, coord2):
    coord_split1 = coord1.split(',')
    coord_split2 = coord2.split(',')

    x11 = int(coord_split1[0])
    y11 = int(coord_split1[1])
    x12 = int(coord_split1[2])
    y12 = int(coord_split1[3])

    x21 = int(coord_split2[0])
    y21 = int(coord_split2[1])
    x22 = int(coord_split2[2])
    y22 = int(coord_split2[3])

    x1_min = min(x11, x12)
    x1_max = max(x11, x12)
    y1_min = min(y11, y12)
    y1_max = max(y11, y12)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)

    x2_min = min(x21, x22)
    x2_max = max(x21, x22)
    y2_min = min(y21, y22)
    y2_max = max(y21, y22)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    cross_x_min = max(x1_min, x2_min)
    cross_x_max = min(x1_max, x2_max)
    cross_y_min = max(y1_min, y2_min)
    cross_y_max = min(y1_max, y2_max)

    cross_width = cross_x_max - cross_x_min
    if cross_width < 0:
        cross_width = 0
    cross_height = cross_y_max - cross_y_min
    if cross_height < 0:
        cross_height = 0

    cross_area = cross_width * cross_height
    union_area = area1 + area2 - cross_area

    iou = 1.0 * cross_area / union_area
    return iou


def calc_tp(submit_df, true_df):
    submit_df = submit_df.sort_values('s', ascending=False, kind='mergesort')
    # print submit_df.head()
    tp = []
    for i, submit_row in submit_df.iterrows():
        # print submit_row
        submit_img_id = submit_row['img_id']
        submit_obj_coordinates = submit_row['bb_coord']

        img_true_objs_df = true_df[true_df.img_id == submit_img_id]
        nearby_true_objs_idxs = []
        for j, true_row in img_true_objs_df.iterrows():
            true_obj_coord = true_row['bb_coord']
            iou = calc_iou(true_obj_coord, submit_obj_coordinates)
            img_true_objs_df.loc[j, 'iou'] = iou
            if iou >= IOU_THRESHOLD:
                nearby_true_objs_idxs.append(j)
        if len(nearby_true_objs_idxs) > 0:
            # print len(tp), 'submit. Found', len(nearby_true_objs_idxs), 'objects!'
            nearby_true_objs_df = img_true_objs_df.loc[nearby_true_objs_idxs, :] \
                .sort_values('iou', ascending=False, kind='mergesort')
            best_fit_row = nearby_true_objs_df.iloc[0]
            best_fit_idx = best_fit_row.name
            # print best_fit_row
            # print best_fit_idx
            true_df = true_df.drop(best_fit_idx)
            tp.append(1)
            # print nearby_true_objs_df
            # print img_true_objs_df
            # print true_df[true_df.img_id == submit_img_id]
            # return
        else:
            tp.append(0)
    return tp


def calc_score(tp, N):
    # for tp_i in tp:
    # print tp_i,
    M = len(tp)
    k_range = range(M)
    r_range = [0]
    p_range = [0]
    z_sum = 0
    Q = 0
    for k in k_range:
        z_sum += tp[k]

        r = z_sum * 1.0 / N
        p = z_sum * 1.0 / (k + 1)

        r_range.append(r)
        p_range.append(p)

        Q += 1.0 / 2 * (p_range[k + 1] + p_range[k]) * (r_range[k + 1] - r_range[k])

    # tp_modified = [-1] + list(tp)
    # print len(tp_modified), len(r_range), len(p_range)
    # vectors_df = pd.DataFrame({'tp': tp_modified, 'rec': r_range, 'prec': p_range})
    # vectors_df.to_csv(os.path.join(os.path.dirname(filepath), 'tp_rec_prec.csv'), sep=';')

    # rating_str = "%.5f" % Q
    # plot(r_range, p_range, filepath, rating_str)
    answer = {'rec': r_range, 'prec': p_range, 'Q': Q}
    return answer


submit_filename = 'PAR1178.csv'
submit_foldername = 'Train'
submit_folderpath = os.path.join('..', submit_foldername)
submit_filepath = os.path.join(submit_folderpath, submit_filename)
submit_df = read(submit_filepath)
print submit_df.head()

true_filepath = os.path.join('..', OBJ_REAL_DIR, OBJ_REAL_FILE)
true_df = read(true_filepath)
print true_df.head()

# set1 = set(true_df[true_df['class'] == 2].img_id)
# print set1
# set1.remove(96)
# print set1
# print 97 in set1, 96 in set1

# print calc_iou('250,400,300,500', '215,416,322,499')
Q = 0
for class_id in CLASSES:
    print class_id
    tp = calc_tp(submit_df[submit_df['class'] == class_id], true_df[true_df['class'] == class_id])
    # print tp
    print true_df[true_df['class'] == class_id].head()
    print sum(tp), len(tp), len(true_df[true_df['class'] == class_id])
    result = calc_score(tp, len(true_df[true_df['class'] == class_id]))
    Q_class = result['Q']
    print Q_class
    Q += Q_class / len(CLASSES)

rating_str = "%.5f" % Q
print
print rating_str

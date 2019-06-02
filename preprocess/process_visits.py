import os
import sys

sys.path.append('/home/LiZhongYu/data/jht/BaiDuBigData2019/')

from tqdm import tqdm
import numpy as np
from paths import train_file_pre_npy_path, test_file_pre_npy_path
from paths import train_visits_274_npy_path, train_visits_224_npy_path
from paths import test_visits_274_npy_path, test_visits_224_npy_path
from paths import train_visit_path, test_visit_path

import datetime
from datetime import datetime as dt

date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

holidays = (
    '20190128', '20190129', '20190130', '20190131', '20190201', '20190202', '20190203',
    '20190204', '20190205', '20190206', '20190207', '20190208', '20190209', '20190210',
    '20181001', '20181002', '20181003', '20181004', '20181005', '20181006', '20181007'
)

# key: 日期   value:  [是否是节假日， 周几]
date2holiday_weekday = {}

# key: 日期   value:  [是否是节假日， 是否是周末]
date2holiday_weekend = {}

# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date = date.__str__().replace("-", "")

    is_holiday = 1 if date in holidays else 0
    weekday = dt.strptime(date, "%Y%m%d").weekday()
    is_weekend = 1 if weekday > 4 else 0

    date2holiday_weekday[date] = [is_holiday, weekday]
    date2holiday_weekend[date] = [is_holiday, is_weekend]

    date_int = int(date)
    date2position[date_int] = [i % 7, i // 7]
    datestr2dateint[str(date_int)] = date_int


def process_txt(lines):
    visit = np.zeros((26, 24, 7))
    # 这里的4 表示一天4个时段，0-5，6-11，12-17，18-23
    visit_274 = np.zeros((2, 7, 4))
    visit_224 = np.zeros((2, 2, 4))

    for line in lines:
        line = line.split()[1]
        # temp = []
        # for item in line.split(','):
        #     temp.append([item[0:8], item[9:].split("|")])
        temp = [[item[0:8], item[9:].split("|")] for item in line.split(',')]

        for date, hour_lst in temp:
            # x - 第几天
            # y - 第几周
            # z - 几点钟
            # value - 到访的总人数
            x, y = date2position[datestr2dateint[date]]
            is_holiday, weekday = date2holiday_weekday[date]
            is_holiday, is_weekend = date2holiday_weekend[date]
            for hour in hour_lst:
                # 统计到访的总人数
                visit[y][str2int[hour]][x] += 1
                visit_274[is_holiday][weekday][int(hour) // 6] += 1
                visit_224[is_holiday][is_weekend][int(hour) // 6] += 1

    visit_274[1] /= 3.0
    visit_274[0] /= 23.0
    visit_224[1] /= 3.0
    visit_224[0] /= 23.0
    visit_224[:, 0, :] /= 5.0
    visit_224[:, 1, :] /= 2.0

    visit_all = (visit, visit_274, visit_224)
    return visit_all


def load(file_pre_npy_path, visit_path):
    visits = []
    visits_274 = []
    visits_224 = []

    file_pre = np.load(file_pre_npy_path)
    for i in tqdm(range(len(file_pre))):
        fname = file_pre[i] + '.txt'
        fname_path = os.path.join(visit_path, fname)
        with open(fname_path) as f:
            lines = f.readlines()
            visit_all = process_txt(lines)

            visits.append(visit_all[0])
            visits_274.append(visit_all[1])
            visits_224.append(visit_all[2])

    visits = np.stack(visits)
    visits_274 = np.stack(visits_274)
    visits_224 = np.stack(visits_224)
    visits_all = (visits, visits_274, visits_224)

    return visits_all


def load_visits(file_pre_npy_path, visits_npy_path, visits_274_npy_path, visits_224_npy_path, visit_path):
    if os.path.exists(visits_274_npy_path) and os.path.exists(visits_224_npy_path) \
            and os.path.exists(visits_npy_path):
        visits, visits_274, visits_224 = np.load(visits_npy_path), \
                                         np.load(visits_274_npy_path), np.load(visits_224_npy_path)
        visits_all = (visits, visits_274, visits_224)
    else:
        visits_all = load(file_pre_npy_path, visit_path)

        np.save(visits_npy_path, visits_all[0])
        np.save(visits_274_npy_path, visits_all[1])
        np.save(visits_224_npy_path, visits_all[2])

    return visits_all


def main():
    train_args = dict(
        file_pre_npy_path=train_file_pre_npy_path,
        visits_npy_path=train_visit_path,
        visits_274_npy_path=train_visits_274_npy_path,
        visits_224_npy_path=train_visits_224_npy_path,
        visit_path=train_visit_path
    )

    test_args = dict(
        file_pre_npy_path=test_file_pre_npy_path,
        visits_npy_path=test_visit_path,
        visits_274_npy_path=test_visits_274_npy_path,
        visits_224_npy_path=test_visits_224_npy_path,
        visit_path=test_visit_path
    )

    train_visits_all = load_visits(**train_args)
    for i in range(len(train_visits_all)):
        print(train_visits_all[i].shape)

    test_visits_all = load_visits(**test_args)
    for i in range(len(test_visits_all)):
        print(test_visits_all[i].shape)


if __name__ == "__main__":
    main()

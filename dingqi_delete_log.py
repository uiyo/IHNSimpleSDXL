import os
import shutil
from datetime import datetime, timedelta

# 定义 base_dir
base_dir = './outputs'

# 获取当前时间
current_time = datetime.now()

# 计算三个月前的日期
three_months_ago = current_time - timedelta(days=90)

# 遍历 base_dir 下的所有人名文件夹
for person_folder in os.listdir(base_dir):
    person_path = os.path.join(base_dir, person_folder)
    
    # 确保是一个文件夹
    if os.path.isdir(person_path):
        # 遍历每个人名文件夹中的所有日期文件夹
        for date_folder in os.listdir(person_path):
            date_path = os.path.join(person_path, date_folder)
            
            # 确保是一个文件夹且符合 yyyy-mm-dd 格式
            try:
                date = datetime.strptime(date_folder, '%Y-%m-%d')
                if date < three_months_ago:
                    # 删除三个月前的文件夹
                    shutil.rmtree(date_path)
                    print(f"已删除：{date_path}")
            except ValueError:
                # 如果日期格式不正确，则跳过
                continue

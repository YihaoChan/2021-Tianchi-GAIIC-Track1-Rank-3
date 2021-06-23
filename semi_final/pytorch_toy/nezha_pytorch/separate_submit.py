import directory


def concat_task_submit():
    """
    拼接两个任务的预测文件
    """
    with open(directory.REGION_RESULT_PATH, 'r') as region_file:
        region_pre = region_file.read()

    with open(directory.CATEGORY_RESULT_PATH, 'r') as category_file:
        category_pre = category_file.read()

    # 检查传入的文件行数是否相同，且是否为17维的区域在前、12维的类型在后
    region_list = region_pre.split('\n')
    category_list = category_pre.split('\n')

    region_num_lines = len(region_list)
    category_num_lines = len(category_list)

    if region_num_lines != category_num_lines:
        raise ValueError('The size of two files is wrong!')

    region_dim = len(region_list[0].split('|')[2].split(' '))
    category_dim = len(category_list[0].split('|')[2].split(' '))

    if region_dim != 17 or category_dim != 12:
        raise ValueError('The dimension of two files is wrong!')

    # 拼接
    test_num = region_num_lines

    submit_res = ''

    with open(directory.SUBMISSION_PATH, 'w') as submit_file:
        for i in range(test_num):
            predecessor = region_list[i]
            successor = category_list[i].split('|')[2]

            submit_res += predecessor + ' ' + successor + '\n'

        submit_res = submit_res.strip('\n')

        submit_file.write(submit_res)


def main():
    concat_task_submit()


if __name__ == '__main__':
    main()

# 读取文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return lines


# 统计拼音出现情况
def count_pinyin_occurrences(new_pinyin_list, original_pinyin_set):
    appeared_count = 0
    not_appeared = []

    for pinyin in new_pinyin_list:
        if pinyin in original_pinyin_set:
            appeared_count += 1
        else:
            not_appeared.append(pinyin)

    return appeared_count, not_appeared


# 主函数
def main():
    # 读取拼音列表
    new_pinyin_list = read_file('pinyin_tone.txt')
    original_pinyin_list = read_file('pinyin_vocab_list_tone.txt')

    # 将原始拼音列表转换为集合以提高查找效率
    original_pinyin_set = set(original_pinyin_list)

    # 统计拼音出现情况
    appeared_count, not_appeared = count_pinyin_occurrences(new_pinyin_list, original_pinyin_set)

    # 打印结果
    print(f"在原始列表中出现的拼音个数: {appeared_count}")
    print(f"没有出现在原始列表中的拼音: {not_appeared}")


if __name__ == "__main__":
    main()

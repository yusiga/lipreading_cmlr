def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return lines

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + '\n')

def reorder_pinyin(pinyin_list):
    group_count = 5
    grouped_pinyin = [[] for _ in range(group_count)]

    for i, pinyin in enumerate(pinyin_list):
        group_index = i % group_count
        grouped_pinyin[group_index].append(pinyin)
    # print(grouped_pinyin[0])
    reordered_list = []
    for j in range(group_count):
        reordered_list+=grouped_pinyin[j]
    # print(reordered_list)
    return reordered_list

def main():
    input_file = 'pinyin_tone.txt'
    output_file = 'pinyin_tone_new.txt'

    pinyin_list = read_file(input_file)
    reordered_pinyin = reorder_pinyin(pinyin_list)
    write_file(output_file, reordered_pinyin)

if __name__ == '__main__':
    main()